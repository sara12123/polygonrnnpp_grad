# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from models.resnet import ResNet, Bottleneck
import torchvision.transforms as transforms
class FPN(nn.Module):
    def __init__(self, final_dim=128):
        super(FPN, self).__init__()
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.final_dim = final_dim

        self.resnet = ResNet(Bottleneck, layers=[3, 4, 6, 3],
                             strides=[1, 2, 1, 1],
                             dilations=[1, 1, 2, 4])

        self.res4_conv = nn.Conv2d(in_channels=2048,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   bias=True)
        self.res3_conv = nn.Conv2d(in_channels=1024,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   bias=True)
        self.res2_conv = nn.Conv2d(in_channels=512,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   bias=True)
        self.res1_conv = nn.Conv2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   bias=True)

        self.topdown4_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.topdown3_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.topdown3_bn = nn.BatchNorm2d(256)
        self.topdown2_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.topdown2_bn = nn.BatchNorm2d(256)
        self.topdown1_bn = nn.BatchNorm2d(256)
        # 上面features的conv
        self.conv_final2 = nn.Conv2d(in_channels=256,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)
        self.bn_final2 = nn.BatchNorm2d(128)

        # 下面features的conv
        self.conv_final1 = nn.Conv2d(in_channels=256,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1,
                                     stride=2,
                                     bias=True)
        self.bn_final1 = nn.BatchNorm2d(128)


    # 逐sample归一化
    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))
        return torch.stack(out, dim=0)

    def forward(self, img):
        """

        :param img:
        :return: feature2: used for 28*28 grid; feature1: used for delta
                features2: [N, 128, 28, 28]
                features1: [N, 128, 14, 14]
        """
        img = self.normalize(img)
        out1, out2, out3, out4 = self.resnet(img)

        # print('out1.shape:', out1.shape)
        # print('out2.shape:', out2.shape)
        # print('out3.shape:', out3.shape)
        # print('out4.shape:', out4.shape)

        top = self.res4_conv(out4)  # (N, 256, 7, 7)
        # print(top.shape)
        features1 = self.res3_conv(out3) + self.topdown4_conv(top)  # (N, 256, 14, 14)
        top2 = self.topdown3_bn(features1)

        tmp = self.res2_conv(out2) + self.topdown3_conv(top2)
        tmp = self.topdown2_bn(tmp)

        features2 = self.res1_conv(out1) + self.topdown2_conv(tmp)  # (N, 256, 56, 56)
        features2 = self.topdown1_bn(features2)
        features2 = self.conv_final1(features2)
        features2 = self.bn_final1(features2)

        features1 = self.conv_final2(top2)
        features1 = self.bn_final2(features1)

        return features2, features1
