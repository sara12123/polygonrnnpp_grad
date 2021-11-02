# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.optim as optim
from polygon_model import PolygonModel
from utils import *
from dataloader import loadData
from losses import delta_loss
import warnings
import torch.nn as nn
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')


'''
CUDA_VISIBLE_DEVICES=5 python train_delta.py
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 train_delta.py

'''

def train(config, load_resnet50=False, pre_trained=None, cur_epochs=0):
    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epoch']

    # train_dataloader = loadData('train', 16, 71, batch_size)
    # val_loader = loadData('val', 16, 71, batch_size, shuffle=False)
    
    train_dataloader = loadData(config['train']['img_path'], config['train']['anno_file'], 16, 71, 'train', batch_size)
    val_loader       = loadData(config['val']['img_path'],   config['val']['anno_file'],   16, 71, 'val',   batch_size)
    model = PolygonModel(load_predtrained_resnet50=load_resnet50,
                         predict_delta=True).cuda()
    # checkpoint
    if pre_trained is not None:
        model.load_state_dict(torch.load(pre_trained))
        print('loaded pretrained polygon net!')

    
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
                weight_decay=config['weight_decay'],
                amsgrad=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config['lr_decay'][0],
                                          gamma=config['lr_decay'][1])
    
    print('Total Epochs:', epochs)
    for it in range(cur_epochs, epochs):
        accum = defaultdict(float)
        # accum['loss_total'] = 0.
        # accum['loss_lstm'] = 0.
        # accum['loss_delta'] = 0.
        for index, batch in enumerate(train_dataloader):
            img = torch.tensor(batch[0], dtype=torch.float).cuda()
            bs = img.shape[0]
            pre_v2 = torch.tensor(batch[2], dtype=torch.float).cuda()
            pre_v1 = torch.tensor(batch[3], dtype=torch.float).cuda()
            outdict = model(img, pre_v2, pre_v1, mode='train_ce')  # (bs, seq_len, 28*28+1)s

            out = outdict['logits']
            out = torch.nn.functional.log_softmax(out, dim=-1)  # logits->log_probs
            out = out.contiguous().view(-1, 28 * 28 + 1)  # (bs*seq_len, 28*28+1)
            target = batch[4]

            # smooth target
            target = dt_targets_from_class(np.array(target, dtype=np.int), 28, 2)  # (bs, seq_len, 28*28+1)
            target = torch.from_numpy(target).cuda().contiguous().view(-1, 28 * 28 + 1)  # (bs, seq_len, 28*28+1)
            # 交叉熵损失计算
            mask_final = batch[6]
            mask_final = torch.tensor(mask_final).cuda().view(-1)
            mask_delta = batch[7]
            mask_delta = torch.tensor(mask_delta).cuda().view(-1)  # (bs*70)
            loss_lstm = torch.sum(-target * torch.nn.functional.log_softmax(out, dim=1), dim=1)  # (bs*seq_len)
            loss_lstm = loss_lstm * mask_final.type_as(loss_lstm)
            loss_lstm = loss_lstm.view(bs, -1)  # (bs, seq_len)
            loss_lstm = torch.sum(loss_lstm, dim=1)  # sum over seq_len  (bs,)
            real_pointnum = torch.sum(mask_final.contiguous().view(bs, -1), dim=1)
            loss_lstm = loss_lstm / real_pointnum  # mean over seq_len
            loss_lstm = torch.mean(loss_lstm)  # mean over batch


            delta_target = prepare_delta_target(outdict['pred_polys'], torch.tensor(batch[-2]).cuda())
            delta_target = dt_targets_from_class(np.array(delta_target.cpu().numpy(), dtype=np.int), 15, 2)  # (bs, seq_len, 14*14+1)
            delta_target = torch.from_numpy(delta_target[:, :, :-1]).cuda().contiguous().view(-1, 15*15)

            delta_logits = outdict['delta_logits'][:, :-1, :]  # (bs, 70, 225)
            delta_logits = delta_logits.contiguous().view(-1, 15*15)  # (bs*70, 225)

            # loss_delta = delta_loss(delta_target, delta_logits, mask_delta)  # -1 去除EoS
            # TODO:get delta loss
            tmp = torch.sum(-delta_target * torch.nn.functional.log_softmax(delta_logits, dim=1), dim=1)
            tmp = tmp * mask_delta.type_as(tmp)
            tmp = tmp.view(bs, -1)
            # sum over seq_len  (bs,)
            tmp = torch.sum(tmp, dim=1)
            real_pointnum2 = torch.sum(mask_delta.contiguous().view(bs, -1), dim=1)
            tmp = tmp / real_pointnum2
            loss_delta = torch.mean(tmp)

            loss = loss_lstm + loss_delta

            model.zero_grad()

            if 'grid_clip' in config:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            loss.backward()

            accum['loss_total'] += loss
            accum['loss_lstm'] += loss_lstm
            accum['loss_delta'] += loss_delta


            optimizer.step()

            # 打印损失
            if (index+1) % 20 == 0:
                print('Epoch {} - Step {}, loss_total {}, loss_lstm {}, loss_delta {}'.format(
                    it + 1,
                    index,
                    accum['loss_total']/20,
                    accum['loss_lstm']/20,
                    accum['loss_delta']/20))
                accum = defaultdict(float)
            # 每3000step一次
            if (index+1) % config['val_every'] == 0:
                # validation
                import pdb;pdb.set_trace()
                model.encoder.eval()
                val_IoU = []
                less_than2 = 0
                with torch.no_grad():
                    for val_index, val_batch in enumerate(val_loader):
                        img = torch.tensor(val_batch[0], dtype=torch.float).cuda()
                        bs = img.shape[0]
                        WH = val_batch[-1]  # WH_dict
                        left_WH = WH['left_WH']
                        origion_WH = WH['origion_WH']
                        object_WH = WH['object_WH']
                        # target，在224*224中的坐标
                        val_target = val_batch[-2].numpy()  # (bs, 70, 2)
                        val_mask_final = val_batch[7]  # (bs, 70)
                        out_dict = model(img, mode='test')  # (N, seq_len) # test_time
                        pred_x = out_dict['final_pred_x'].cpu().numpy()
                        pred_y = out_dict['final_pred_y'].cpu().numpy()

                        pred_len = out_dict['lengths']
                        # 求IoU
                        for ii in range(bs):
                            vertices1 = []
                            vertices2 = []
                            scaleW = 224.0 / object_WH[0][ii]
                            scaleH = 224.0 / object_WH[1][ii]
                            leftW = left_WH[0][ii]
                            leftH = left_WH[1][ii]

                            all_len = np.sum(val_mask_final[ii].numpy())
                            cnt_target = val_target[ii][:all_len]
                            for vert in cnt_target:
                                vertices2.append((vert[0]/scaleW + leftW,
                                                  vert[1]/scaleH + leftH))

                            # print('target:', cnt_target)
                            pred_len_b = pred_len[ii] - 1
                            if pred_len_b < 2:
                                val_IoU.append(0.)
                                less_than2 += 1
                                continue

                            for j in range(pred_len_b):
                                vertex = (
                                    pred_x[ii][j] / scaleW + leftW,
                                    pred_y[ii][j] / scaleH + leftH
                                )
                                vertices1.append(vertex)

                            _, nu_cur, de_cur = iou(vertices1, vertices2, origion_WH[1][ii], origion_WH[0][ii])  # (H, W)
                            iou_cur = nu_cur * 1.0 / de_cur if de_cur != 0 else 0
                            val_IoU.append(iou_cur)

                val_iou_data = np.mean(np.array(val_IoU))
                print('Validation After Epoch {} - step {}'.format(str(it + 1), str(index + 1)))
                print('           IoU      on validation set: ', val_iou_data)
                print('less than 2: ', less_than2)
                if it > 5:  # it = 6
                    print('Saving training parameters after this epoch:')
                    torch.save(model.state_dict(),
                               '/workdir/FPN_Epoch{}-Step{}_ValIoU{}.pth'.format(
                                   str(it + 1),
                                   str(index + 1),
                                   str(val_iou_data)))
                # set to init
                model.train()  # important

        # 衰减
        scheduler.step()
        print()
        print('Epoch {} Completed!'.format(str(it+1)))
        print()

if __name__ == '__main__':
    config = {}
    config['batch_size'] = 8 #8
    config['lr'] = 0.0001
    config['num'] = 16
    # epochs over the whole dataset
    config['epoch'] = 20
    config['lr_decay'] = [5, 0.1]
    config['weight_decay'] = 0.00001
    config['grad_clip'] = 40
    config['val_every'] = 3000

    config['train'] = {'img_path':'/home/zhangmingming_2020/data/building/building_coco/train/images',
                        'anno_file': '/home/zhangmingming_2020/data/building/building_coco/annotation/train.json'}
    
    config['val'] = {'img_path':'/home/zhangmingming_2020/data/building/building_coco/val/images',
                        'anno_file': '/home/zhangmingming_2020/data/building/building_coco/annotation/val.json'}

    train(config, load_resnet50=True)
