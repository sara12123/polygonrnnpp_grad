# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1+(dilation-1), bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差块
        residule = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 残差块输入x 维度调整
        if self.downsample is not None:
            residule = self.downsample(x)

        out = out + residule
        out = self.relu(out)
        return out

# resnet-50
class ResNet(nn.Module):
    def __init__(self, block, layers, strides, dilations, num_classes=1000):
        """

        :param block: 残差块对象bottlenneck
        :param layers: 每个残差块重复次数列表 [3, 4, 6, 3]
        :param strides: 步长 list，即每个残差块3*3卷积的步长[1, 2, 1, 1]
        :param dilations: 膨胀因子 list, 1表示正常卷积， 空洞卷积扩大感受野, [1, 1, 2, 4]
        :param num_classes: 1000
        """
        super(ResNet, self).__init__()
        self.inplanes = 64  # 输入维度
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # out: (N, 64, 112, 112)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # out:(N, 64, 56, 56)

        # layer1: 输入: (N, 64, 56, 56) 输出: (N, 256, 56, 56)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=strides[0], dilation=dilations[0])
        # 只有第二层步长为2，为了调整layer2的输入x->分辨率/2,->适应输出的分辨率变为28:之后一直维持28不变
        # 这点与原paper不同
        # layer2: 输入：(N, 256, 56, 56) 输出：(N, 512, 28, 28)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=strides[1], dilation=dilations[1])
        # layer3: 输出(N, 512, 28, 28) 输出：(N, 1024, 28, 28), stride=1, 空洞=2
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=strides[2], dilation=dilations[2])
        # out: (N, 512, 28, 28)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=strides[3], dilation=dilations[3])
        # out: (N, 2048, 28, 28)

        # 全局池化kernel_size=28
        self.avgpool = nn.AvgPool2d(7 * max(dilations), stride=1)
        # out: (N, 512*4, 1, 1)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        # out: (N, 1000)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_f = self.relu(x)
        x = self.maxpool(conv1_f)

        # 与resnet原paper的不同就是，原paperlayer2/3/4后都加了一层max_pool with stride=2(down sampling)
        layer1_f = self.layer1(x)  # out: (N, 256, 56, 56) res1
        out1 = layer1_f
        layer2_f = self.layer2(layer1_f)  # out: (N, 512, 28, 28) res2
        out2 = layer2_f
        layer3_f = self.layer3(layer2_f)  # out: (N, 1024, 28, 28)  res3
        # out3 = self.maxpool(layer3_f)  # 14
        out3 = self.max2(layer3_f)
        layer4_f = self.layer4(out3)  # out: (N, 2048, 28, 28)  res4
        # out4 = self.maxpool(layer4_f)  # 7
        out4 = self.max2(layer4_f)

        # x = self.avgpool(layer4_f)
        # x = x.view(x.size(0), -1)
        # fc_f = self.fc(x)

        # return fc_f, conv1_f, layer1_f, layer2_f, layer3_f, layer4_f
        return out1, out2, out3, out4

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """

        :param block: bottleneck残差块对象(类引用)
        :param planes: 输入channel
        :param blocks: 残差块的重复次数: 3/4/6/3
        :param stride: 步长 [1, 2, 1, 1]
        :param dilation: 膨胀因子 1/1/2/4
        :return:
        """
        downsample = None
        # 需要对输入残差块x进行channel数量调整
        if stride!=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        # 更新 inplanes
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)



if __name__=='__main__':
    res = ResNet(Bottleneck, layers=[3,4,6,3], strides=[1,2,1,1], dilations=[1,1,2,4])
    res.load_state_dict(torch.load('resnet50.pth'))

