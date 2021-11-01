# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from models.polyrnnpp import PolyRNNpp
from utils.utils import *
from loadData import loadData
import warnings

warnings.filterwarnings('ignore')


def train(config, load_resnet50=False, pre_trained=None, cur_epochs=0):

    devices = config['gpu_id']
    torch.cuda.set_device(devices[0])
    batch_size = config['batch_size']
    lr = config['lr']
    # 总共的迭代轮数~
    epochs = config['epoch']
    # dataloader
    # 以前写的seq_len=70, 可能得看一下改成71行不行
    train_dataloader = loadData('train', 0, 71, batch_size)
    val_loader = loadData('val', 0, 71, batch_size, shuffle=False)
    model = PolyRNNpp(load_predtrained_resnet50=load_resnet50)
    # checkpoint
    if pre_trained is not None:
        model.load_state_dict(torch.load(pre_trained))
        print('loaded pretrained polygon net!')

    # Regulation，原paper没有+regulation
    no_wd = []
    wd = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # No optimization for frozen params
            continue
        if 'bn' in name or 'convLSTM' in name or 'bias' in name:
            no_wd.append(param)
        else:
            wd.append(param)

    optimizer = optim.Adam(
                [
                    {'params': no_wd, 'weight_decay': 0.0},
                    {'params': wd}
                ],
                lr=lr,
                weight_decay=0.00001,
                amsgrad=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=6,
                                          gamma=0.5)

    print('total epochs:', epochs)
    for it in range(cur_epochs, epochs):  # 整个数据集上的轮次数
        model.train()
        train_IoU = []
        train_acc = []
        for index, batch in enumerate(train_dataloader):
            img = torch.tensor(batch[0], dtype=torch.float).to(devices)
            bs = img.shape[0]
            pre_v2 = torch.tensor(batch[2], dtype=torch.float).cuda()
            pre_v1 = torch.tensor(batch[3], dtype=torch.float).cuda()

            outdict = model(img, pre_v2, pre_v1, mode='train_ce')  # (bs, seq_len, 28*28+1)s

            out = outdict['logits']
            out = torch.nn.functional.log_softmax(out, dim=-1)  # logits->log_probs
            out = out.contiguous().view(-1, 28 * 28 + 1)  # (bs*seq_len, 28*28+1)
            target = batch[4]

            target_accuracy = torch.tensor(target, dtype=torch.long).cuda().contiguous().view(-1)
            # smooth target
            target = dt_targets_from_class(np.array(target, dtype=np.int), 28, 2)  # (bs, seq_len, 28*28+1)
            # print(ta.shape)  # (16, 69, 28*28+1)
            target = torch.from_numpy(target).cuda().contiguous().view(-1, 28 * 28 + 1)  # (bs, seq_len, 28*28+1)

            # 交叉熵损失计算
            mask_final = batch[6]  # 结束符标志mask  (bs, seq_len(70)从第一个点开始)
            mask_final = torch.tensor(mask_final).cuda().contiguous().view(-1)

            loss_lstm = torch.sum(-target * torch.nn.functional.log_softmax(out, dim=1), dim=1)  # (bs*seq_len)
            loss_lstm = loss_lstm * mask_final.type_as(loss_lstm)  # 从end point截断损失计算
            loss_lstm = loss_lstm.view(bs, -1)  # (bs, seq_len)
            loss_lstm = torch.sum(loss_lstm, dim=1)  # sum over seq_len  (bs,)
            real_pointnum = torch.sum(mask_final.contiguous().view(bs, -1), dim=1)
            loss_lstm = loss_lstm / real_pointnum  # mean over seq_len
            loss_lstm = torch.mean(loss_lstm)  # mean over batch
            model.zero_grad()
            loss_lstm.backward()
            optimizer.step()

            # train_accuracy
            result_index = torch.argmax(out, dim=1)  # (bs*seq_len)
            tmp = torch.tensor(result_index == target_accuracy, dtype=torch.float).cuda()
            accuracy = (tmp * mask_final).sum().item()
            accuracy = accuracy * 1.0 / mask_final.sum().item()
            train_acc.append(accuracy)

            # 打印损失
            if index % 100 == 0:
                print('Epoch {} - Iteration {}, loss {}, acc {}'.format(it + 1, index, loss_lstm.data, accuracy))
        # 衰减
        scheduler.step()

        # val
        model.eval()
        val_IoU = []
        val_acc = []
        with torch.no_grad():
            for val_index, val_batch in enumerate(val_loader):
                img = torch.tensor(val_batch[0], dtype=torch.float).cuda()
                bs = img.shape[0]
                seq_len = 71
                # pre_v2 = torch.tensor(val_batch[2], dtype=torch.float).cuda()
                # pre_v1 = torch.tensor(val_batch[3], dtype=torch.float).cuda()
                # padding 0 for pre_v2, pre_v1
                pre_v2 = torch.zeros((bs, seq_len-2, img.shape[-1]))
                pre_v1 = torch.zeros((bs, seq_len-1, img.shape[-1]))
                val_mask_final = val_batch[6]
                val_mask_final = torch.tensor(val_mask_final).cuda().contiguous().view(-1)

                out_dict = model(img, pre_v2, pre_v1, mode='test')  # (N, seq_len) # test_time
                pred_polys = out_dict['pred_polys']  # (bs, seq_len)
                tmp = pred_polys
                pred_polys = pred_polys.contiguous().view(-1)  # (bs*seq_len)
                val_target = val_batch[4]  # (bs, seq_len)
                # 求accuracy
                val_target = torch.tensor(val_target, dtype=torch.long).cuda().contiguous().view(-1)  # (bs*seq_len)
                val_acc1 = torch.tensor(pred_polys == val_target, dtype=torch.float).cuda()
                val_acc1 = (val_acc1 * val_mask_final).sum().item()
                val_acc1 = val_acc1 * 1.0 / val_mask_final.sum().item()
                val_acc.append(val_acc1)
                # 用作计算IoU
                val_result_index = tmp.cpu().numpy()  # (bs, seq_len)
                val_target = val_batch[4].numpy()  # (bs, seq_len)
                less_than2 = 0
                # 求IoU
                for ii in range(bs):
                    vertices1 = []
                    vertices2 = []

                    for label in val_result_index[ii]:
                        if label == 28 * 28:
                            break
                        vertex = (
                            ((label % 28) * 8.0 + 4),
                            ((int(label / 28)) * 8.0 + 4))
                        vertices1.append(vertex)
                    for label in val_target[ii]:
                        if label == 28 * 28:
                            break
                        vertex = (
                            ((label % 28) * 8.0 + 4),
                            ((int(label / 28)) * 8.0 + 4))
                        vertices2.append(vertex)
                    if len(vertices1) < 2 or len(vertices2) < 2:
                        less_than2 += 1
                        continue
                    _, nu_cur, de_cur = iou(vertices1, vertices2, 224, 224)
                    iou_cur = nu_cur * 1.0 / de_cur if de_cur != 0 else 0
                    val_IoU.append(iou_cur)


        val_acc_data = np.mean(np.array(val_acc))
        val_iou_data = np.mean(np.array(val_IoU))
        train_acc = np.mean(np.array(train_acc))
        print('Validation After Epoch {}'.format(str(it + 1)))
        print('           Accuracy on training   set: ', train_acc)
        print('           Accuracy on validation set: ', val_acc_data)
        print('           IoU      on validation set: ', val_iou_data)

        if (it + 1) % 10 == 0:
            print('Saving training parameters after this epoch:')
            torch.save(model.state_dict(),
                       '/data/duye/pretrained_models/PolyRNNpp_lr0.0001_Epoch{}_ValIoU{}.pth'.format(
                           str(it + 1),
                           str(val_iou_data)))

        print('Epoch {} Completed!'.format(str(it+1)))

if __name__ == '__main__':
    config = {}
    config['gpu_id'] = [9]
    config['batch_size'] = 16  # 有一篇paper说最好BatchSize<=32, 原作者设置batch_size=8
    # 适当加大下学习率应该是有用的, 0.0005貌似更好用一些
    config['lr'] = 0.0001
    config['num'] = 16
    # epochs over the whole dataset
    config['epoch'] = 30
    train(config, load_resnet50=True)
