# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from FPN import FPN
from convLSTM import AttConvLSTM

class PolygonModel(nn.Module):
    def __init__(self,
                 load_predtrained_resnet50=False,
                 predict_delta=False):
        super(PolygonModel, self).__init__()
        self.load_predtrained_resnet50 = load_predtrained_resnet50
        self.predict_delta = predict_delta

        self.encoder = FPN()
        self.decoder = AttConvLSTM(predict_delta=True)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        res_path = '/home/zhangmingming_2020/data/resnet/resnet50-19c8e357.pth'
        if load_predtrained_resnet50:
            self.encoder.resnet.load_state_dict(torch.load(res_path))
            print('Load pretrained resnet50 completed!')

    def forward(self, img,
                pre_v2=None,
                pre_v1=None,
                temperature=0.0,
                mode='train_ce',
                fp_beam_size=1,
                beam_size=1,
                return_attention=False,
                use_correction=False):

        if beam_size == 1:
            if mode == 'train_ce':
                feats, feats_delta = self.encoder(img)
                return self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)

            elif mode == 'train_rl':
                feats, feats_delta = self.encoder(img)
                return self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)

            elif mode == 'test':
                feats, feats_delta = self.encoder(img)
                outdict = self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)

                delta_pred = outdict['delta_pred']  # (bs, len_s=71, ignore EOS)
                pred = outdict['pred_polys']  # (bs, len_s=71)

                dx = delta_pred % 15 - 7  # (bs, len_s)
                dy = (delta_pred / 15).int() - 7

                # To(0, 112)
                pred_x = (pred % 28) * 4.0 + 2  # (bs, len_s)
                pred_y = (pred / 28).int() * 4.0 + 2  # (bs, len_s)

                # (0,112) +delta 防溢出
                pred_x = pred_x + dx
                pred_y = pred_y + dy

                index1 = (pred_x > 111)
                pred_x[index1] = 111
                index2 = (pred_x < 0)
                pred_x[index2] = 0
                index1 = (pred_y > 111)
                pred_y[index1] = 111
                index2 = (pred_y < 0)
                pred_y[index2] = 0

                # To (0, 224)
                pred_x = pred_x * 2.0 + 1
                pred_y = pred_y * 2.0 + 1

                outdict['final_pred_x'] = pred_x
                outdict['final_pred_y'] = pred_y

                return outdict

        else:
            return None

    def append(self, module):
        if module == 'delta_module':
            self.decoder.predict_delta = True
            self.decoder.delta_downsample_conv = nn.Conv2d(in_channels=17,
                                                       out_channels=17,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       bias=True)
            self.decoder.delta_conv1 = nn.Conv2d(in_channels=128 + 17,
                                             out_channels=64,
                                             kernel_size=3,
                                             padding=1,
                                             stride=1,
                                             bias=True)

            # init
            nn.init.kaiming_normal_(self.decoder.delta_conv1.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_conv1.bias is not None:
                nn.init.constant_(self.decoder.delta_conv1.bias, 0)

            self.decoder.delta_bn1 = nn.BatchNorm2d(64)
            # init
            nn.init.constant_(self.decoder.delta_bn1.weight, 1)
            nn.init.constant_(self.decoder.delta_bn1.bias, 0)

            self.decoder.delta_res1_conv = nn.Conv2d(in_channels=128 + 17,
                                                 out_channels=64,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=True)

            nn.init.kaiming_normal_(self.decoder.delta_res1_conv.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_res1_conv.bias is not None:
                nn.init.constant_(self.decoder.delta_res1_conv.bias, 0)

            self.decoder.delta_res1_bn = nn.BatchNorm2d(64)
            # init
            nn.init.constant_(self.decoder.delta_res1_bn.weight, 1)
            nn.init.constant_(self.decoder.delta_res1_bn.bias, 0)

            self.decoder.delta_relu1 = nn.ReLU(inplace=True)  # out: (N, 64, 14, 14)

            self.decoder.delta_conv2 = nn.Conv2d(in_channels=64,
                                             out_channels=16,
                                             kernel_size=3,
                                             padding=1,
                                             stride=1,
                                             bias=True)

            # init
            nn.init.kaiming_normal_(self.decoder.delta_conv2.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_conv2.bias is not None:
                nn.init.constant_(self.decoder.delta_conv2.bias, 0)

            self.decoder.delta_bn2 = nn.BatchNorm2d(16)
            # init
            nn.init.constant_(self.decoder.delta_bn2.weight, 1)
            nn.init.constant_(self.decoder.delta_bn2.bias, 0)

            self.decoder.delta_res2_conv = nn.Conv2d(in_channels=64,
                                                 out_channels=16,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=True)
            # init
            nn.init.kaiming_normal_(self.decoder.delta_res2_conv.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_res2_conv.bias is not None:
                nn.init.constant_(self.decoder.delta_res2_conv.bias, 0)

            self.decoder.delta_res2_bn = nn.BatchNorm2d(16)
            nn.init.constant_(self.decoder.delta_res2_bn.weight, 1)
            nn.init.constant_(self.decoder.delta_res2_bn.bias, 0)


            self.decoder.delta_relu2 = nn.ReLU(inplace=True)

            self.decoder.delta_final = nn.Linear(16 * 14 * 14, 14 * 14)  # 输出一个14*14的grid
            # init
            nn.init.xavier_uniform_(self.decoder.delta_final.weight)
            nn.init.constant_(self.decoder.delta_final.bias, 0)

            self.decoder.use_delta_attn = True
                # delta attn
            self.decoder.delta_fc_att = nn.Linear(in_features=self.feat_channels,
                                              out_features=1)
            # init
            nn.init.xavier_uniform_(self.decoder.delta_fc_att.weight)
            nn.init.constant_(self.decoder.delta_fc_att.bias, 0)

            self.decoder.delta_conv_att = nn.Conv2d(in_channels=17,
                                                out_channels=self.feat_channels,
                                                kernel_size=1,
                                                padding=0,
                                                bias=True)
            # init
            nn.init.kaiming_normal_(self.decoder.delta_conv_att.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_conv_att.bias is not None:
                nn.init.constant_(self.decoder.delta_conv_att.bias, 0)

