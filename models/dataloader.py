# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from utils import get_edge_mask, poly01_to_poly0g, poly0g_to_poly01
import warnings
warnings.filterwarnings("ignore")

# TODO: 修改dataloader, 构造delta
class CityScape(Dataset):
    # seq_len: default 70
    def __init__(self, num, path, seq_len, transform=None):
        """
        :param num: default 16
        :param path: new_data/train或test或val
        :param length: time step，max vertex num, default 60
        :param transform: ToTensor()
        """
        super(CityScape, self).__init__()
        # 加载数据集
        self.num = num
        self.seq_len = seq_len
        self.path = path  # train{test,val}
        meta = json.load(open(str(path) + '_meta.json'))  # 配置文件在同一目录下
        self.total_num = meta['total_count']
        self.select_classes = meta['select_classes']
        self.transform = transform

    # num pf samples
    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        """
        :param index:
        :return:
            img: FloatTensor(3, 224, 224)
            label_onehot[2]: first vertex, ndarray, onehot, (28*28+3)
            label_onehot[:-2]: yt-2, pre 2-vertex, ndarray, (length, 28*28+1)
            label_onehot[1:-1]: yt-1, pre 1-vertex
            label_index[2:]: GT index in the one-hot vector, used for calculating loss
        """
        def getdata(mode, flip, index, random_start=False):
            """

            :param mode:
            :param flip: 0不反转，1反转
            :param index:
            :return:
            """
            # 打开png和json文件
            try:
                img = Image.open('new_img/' + str(mode) + '/' + str(index) + '.png').convert('RGB')
            except FileNotFoundError:
                return None
            assert not (img is None)

            W = img.width
            H = img.height

            if flip:
                fl = transforms.RandomHorizontalFlip()
                img = fl(img)

            # label为多边形的顶点列表
            js = json.load(open('new_label/' + str(mode) + '/' + str(index) + '.json'))
            label = js['polygon']  # 多边形顶点
            classes = js['label']  # 分类
            left_WH = js['left_WH']  # 裁减图片左上角坐标在原图中的WH
            object_WH = js['object_WH']  # 裁减下来的图片(scale到224,224之前)的WH
            origion_WH = js['origion_WH']  # 原始图片的WH
            WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
            point_num = len(label)
            polygon = np.array(label, dtype=np.float)  # (point_num, 2)

            # 去除重复点
            # To (0,1)
            polygon = polygon / (W*1.0)

            # To (0, g), int值,即在28*28中的坐标值 道格拉斯算法多边形曲线拟合，这里有一个去除重点的过程
            polygon = poly01_to_poly0g(polygon, 28)
            label_onehot = np.zeros([self.seq_len, 28 * 28 + 1])
            label_index = np.zeros([self.seq_len])
            point_scaled = []

            mask_final = np.zeros(self.seq_len, dtype=np.int)
            mask_delta = np.zeros(self.seq_len - 1, dtype=np.int)
            # 随机选取开始点
            if random_start:
                polygon = np.roll(polygon, np.random.choice(range(point_num)), axis=0)

            # 这个其实可以写成向量形式 加速计算
            cnt = 0
            point_num = len(polygon)

            polygon01 = np.zeros([self.seq_len-1, 2])
            tmp = np.array(poly0g_to_poly01(polygon, 28) * W, dtype=np.int)  # To(0, 224)
            if point_num <= 70:
                polygon01[:point_num] = tmp
            else:
                polygon01[:70] = tmp[:70]


            if point_num < self.seq_len:  # < 70
                for point in polygon:
                    # 反转W坐标
                    if flip:
                        x = 27 - point[0]
                    else:
                        x = point[0]
                    y = point[1]

                    indexs = y * 28 + x
                    label_index[cnt] = indexs
                    label_onehot[cnt, indexs] = 1
                    cnt += 1
                    point_scaled.append([x, y])
                # 添加第一个点构成一个循环
                # xf = polygon[0][0]
                # if flip:
                #     xf = 27 - polygon[0][0]
                # yf = polygon[0][1]
                # indexs = yf * 28 + xf
                # label_index[cnt] = indexs
                # label_onehot[cnt, indexs] = 1
                # cnt += 1
                # point_scaled.append([xf, yf])
                mask_final[:cnt+1] = 1  # +1才会计算final最后EOS的损失
                mask_delta[:cnt] = 1
                # end point
                label_index[cnt] = 28 * 28
                label_onehot[cnt, 28 * 28] = 1
                cnt += 1
                for ij in range(cnt, self.seq_len):
                    label_index[ij] = 28 * 28
                    label_onehot[ij, 28 * 28] = 1
                    cnt += 1
            else:
                # 点数过多的话只取前70个点是不对的, 这里应该考虑一下如何选取点
                for iii in range(self.seq_len - 1):
                    point = polygon[iii]  # 取点
                    if flip:
                        x = 27 - point[0]
                    else:
                        x = point[0]
                    y = point[1]
                    xx = x
                    yy = y
                    indexs = yy * 28 + xx
                    label_index[cnt] = indexs
                    label_onehot[cnt, indexs] = 1
                    cnt += 1
                    point_scaled.append([xx, yy])
                # # 构成循环
                # xf = polygon[0][0]
                # if flip:
                #     xf = 27 - polygon[0][0]
                # yf = polygon[0][1]
                # indexs = yf * 28 + xf
                # label_index[cnt] = indexs
                # label_onehot[cnt, indexs] = 1
                # cnt += 1
                # point_scaled.append([xf, yf])
                mask_final[:cnt + 1] = 1   # +1才会计算final最后EOS的损失
                mask_delta[:cnt] = 1
                # EOS
                label_index[self.seq_len-1] = 28 * 28
                label_onehot[self.seq_len-1, 28 * 28] = 1

            # ToTensor
            if self.transform:
                img = self.transform(img)

            point_scaled = np.array(point_scaled)
            # 边界，edge上的点为1，其余点为0
            edge_mask = np.zeros((28, 28), dtype=np.float)
            edge_mask = get_edge_mask(point_scaled, edge_mask)

            # 返回第一个点，yt-2, yt-1
            # label_onehot: (seq_len, 28*28+1)
            # label_index: seq_len
            return img, \
                    label_onehot[0],\
                    label_onehot[:-2], \
                    label_onehot[:-1],\
                    label_index, \
                    edge_mask,\
                    mask_final, \
                    mask_delta, \
                    polygon01, WH
            # polygon01: 在224*224中的坐标, 而非0-1

        if self.path == 'train':
            # flip = np.random.choice(2)  # (0,1)
            flip = 0
            if flip == 0:  # 使用原始图
                return getdata('train', 0, index, random_start=False)  # 先不随机开始
            elif flip == 1:   # 反转
                return getdata('train', 1, index, random_start=False)
        else:
            #  test/val 数据集
            return getdata(self.path, 0, index, random_start=False)

"""
load data
"""
def loadData(path, data_num, len_s, batch_size, shuffle=True):

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform = transforms.Compose([transforms.ToTensor(), ])

    cityscape = CityScape(data_num, path, len_s, transform)
    dataloader = DataLoader(cityscape, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False)

    print('DataLoader complete!', dataloader)
    return dataloader

if __name__ == '__main__':
    loader = loadData('train', 16, 70, 16)
    print(len(loader))
