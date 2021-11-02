# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import random
import numpy as np
from skimage.io import imread
import skimage.color as color
import skimage.transform as skimgtransform
from pycocotools.coco import COCO
from utils import get_edge_mask, poly01_to_poly0g, poly0g_to_poly01
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-7

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


# TODO: 修改dataloader, 构造delta
class Building_del(Dataset):
    # seq_len: default 70
    def __init__(self, num, seq_len, mode, img_path, anno_path, transform=None):
        """
        :param num: default 16
        :param path: new_data/train或test或val
        :param length: time step，max vertex num, default 60
        :param transform: ToTensor()
        """
        super(Building, self).__init__()
        # 加载数据集
        self.num = num
        self.seq_len = seq_len
        self.mode = mode # train{test, val}
        self.img_path = img_path  # images
        self.anno_path = anno_path  # train{test,val}.json
        self.CLASSES = 'building'
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.anno_path)
        # self.select_classes = len(cat_ids)
        self.transform = transform

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        data_infos = []
        total_ann_ids = []
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            
            ann_ids = self.coco.getAnnIds([img_id])
            ann_infos = self.coco.loadAnns(ann_ids)
            for i, ann in enumerate(ann_infos):              
                if ann.get('ignore', False):
                    continue
                if ann.get('iscrowd', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                
                bbox = ann['bbox']
                ann = dict(
                    file_name = img_info['file_name'],
                    img_w = img_info['width'],
                    img_h = img_info['height'],
                    label=self.CLASSES,
                    bbox=bbox,
                    polygon=ann.get('segmentation', None))
                
                data_infos.append(ann)
            
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        
        return data_infos

    # num pf samples
    def __len__(self):
        return len(self.data_infos)
        

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
        def getdata(data_info, flip, random_start=False):
            """

            :param img_path:
            :param flip: 0不反转，1反转
            :param index:
            :return:
            """
            
            # 打开png和json文件
            try:
                img = Image.open(os.path.join(self.img_path, data_info['file_name'])).convert('RGB')
            except FileNotFoundError:
                return None
            assert not (img is None)

            W = img.width
            H = img.height

            if flip:
                fl = transforms.RandomHorizontalFlip()
                img = fl(img)

            classes = data_info['label']  # 分类
            left_WH = data_info['bbox'][:2]  # 裁减图片左上角坐标在原图中的WH
            object_WH = data_info['bbox'][2:]  # 裁减下来的图片(scale到224,224之前)的WH
            origion_WH = (data_info['img_w'], data_info['img_h']) # 原始图片的WH
            WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
            label = data_info['polygon'][0] # label为多边形的顶点列表
            polygon = np.array(label, dtype=np.float).reshape(-1,2)  # (point_num, 2)
            point_num = len(polygon)

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

        if self.mode == 'train':
            # flip = np.random.choice(2)  # (0,1)
            flip = 0
            if flip == 0:  # 使用原始图
                return getdata(self.data_infos[index], 0, random_start=False)  # 先不随机开始
            elif flip == 1:   # 反转
                return getdata(self.data_infos[index], 1, random_start=False)
        else:
            #  test/val 数据集
            return getdata(self.data_infos[index], 0, random_start=False)

# TODO: 修改dataloader, 构造delta
class Building(Dataset):
    # seq_len: default 70
    def __init__(self, num, seq_len, mode, img_path, anno_path, transform=None):
        """
        :param num: default 16
        :param path: new_data/train或test或val
        :param length: time step，max vertex num, default 60
        :param transform: ToTensor()
        """
        super(Building, self).__init__()
        # 加载数据集
        self.num = num
        self.seq_len = seq_len
        self.mode = mode # train{test, val}
        self.img_path = img_path  # images
        self.anno_path = anno_path  # train{test,val}.json
        self.CLASSES = 'building'
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.anno_path)
        self.transform = transform

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        data_infos = []
        total_ann_ids = []
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            
            ann_ids = self.coco.getAnnIds([img_id])
            ann_infos = self.coco.loadAnns(ann_ids)
            for i, ann in enumerate(ann_infos):              
                if ann.get('ignore', False):
                    continue
                if ann.get('iscrowd', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                
                bbox = ann['bbox']
                ann = dict(
                    file_name = img_info['file_name'],
                    img_w = img_info['width'],
                    img_h = img_info['height'],
                    label=self.CLASSES,
                    bbox=bbox,
                    polygon=ann.get('segmentation', None))
                
                data_infos.append(ann)
            
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        
        return data_infos

    # num pf samples
    def __len__(self):
        return len(self.data_infos)
    
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
        if self.mode == 'train':
            # flip = np.random.choice(2)  # (0,1)
            flip = 0
            if flip == 0:  # 使用原始图
                return self.getdata(self.data_infos[index], 0, random_start=False)  # 先不随机开始
            elif flip == 1:   # 反转
                return self.getdata(self.data_infos[index], 1, random_start=False)
        else:
            #  test/val 数据集
            return self.getdata(self.data_infos[index], 0, random_start=False)

    def getdata(self, data_info, flip, random_start=False):
        """
            :param img_path:
            :param flip: 0不反转，1反转
            :param index:
            :return:
        """
        
        lo,hi = (0.1,0.2)
        context_expansion = random.uniform(lo, hi)
        crop_info = self.extract_crop(data_info, context_expansion)
        '''
        crop_info = {
            'img': new_img,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen,
            'poly':poly
        }
        '''
        img = crop_info['img']
        if flip:
            fl = transforms.RandomHorizontalFlip()
            img = fl(img)
        # pdb.set_trace()
        # for Torch, use CHW, instead of HWC
        # img = img.transpose(2,0,1)
        # ToTensor
        if self.transform:
            img = self.transform(img)
        
        '''
            orig_poly = poly.copy()

            # Convert from [0, 1] to [0, grid_side]
            poly = utils.poly01_to_poly0g(poly, self.opts['grid_side'])
            fwd_poly = poly
            fwd_poly = np.append(fwd_poly, [fwd_poly[0]], axis=0)
            bwd_poly = fwd_poly[::-1]
            if flip:
                fwd_poly, bwd_poly = bwd_poly, fwd_poly

            arr_fwd_poly = np.ones((self.opts['max_poly_len'], 2), np.float32) * -1
            arr_bwd_poly = np.ones((self.opts['max_poly_len'], 2), np.float32) * -1
            arr_mask = np.zeros(self.opts['max_poly_len'], np.int32)
            len_to_keep = min(len(fwd_poly), self.opts['max_poly_len'])
            arr_fwd_poly[:len_to_keep] = fwd_poly[:len_to_keep]
            arr_bwd_poly[:len_to_keep] = bwd_poly[:len_to_keep]
            arr_mask[:len_to_keep+1] = 1
            # Numpy doesn't throw an error if the last index is greater than size
            return_dict = {
                'img': img,
                'img_path': instance['img_path'],
                'patch_w': crop_info['patch_w'],
                'starting_point': crop_info['starting_point']
                'fwd_poly': arr_fwd_poly,
                'bwd_poly': arr_bwd_poly,
                'mask': arr_mask,
                'orig_poly': orig_poly,
                'full_poly': fwd_poly,
            }
        '''
        classes = data_info['label']  # 分类
        left_WH = data_info['bbox'][:2]  # 裁减图片左上角坐标在原图中的WH
        object_WH = data_info['bbox'][2:]  # 裁减下来的图片(scale到224,224之前)的WH
        origion_WH = (data_info['img_w'], data_info['img_h']) # 原始图片的WH
        WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}

        H,W = img.shape[1:]
        # label = data_info['polygon'][0] # label为多边形的顶点列表
        polygon = crop_info['poly']# np.array(label, dtype=np.float).reshape(-1,2)  # (point_num, 2)
        point_num = len(polygon)
        
        # 去除重复点
        # To (0,1)
        polygon = polygon / (W*1.0)
        # To (0, g), int值,即在28*28中的坐标值 道格拉斯算法多边形曲线拟合，这里有一个去除重点的过程
        polygon = poly01_to_poly0g(polygon, 28)
        # 随机选取开始点
        if random_start:
            polygon = np.roll(polygon, np.random.choice(range(point_num)), axis=0)

        # 这个其实可以写成向量形式 加速计算
        label_onehot = np.zeros([self.seq_len, 28 * 28 + 1])
        label_index = np.zeros([self.seq_len])
        
        mask_final = np.zeros(self.seq_len, dtype=np.int)
        mask_delta = np.zeros(self.seq_len - 1, dtype=np.int)

        point_scaled = []

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

            mask_final[:cnt + 1] = 1   # +1才会计算final最后EOS的损失
            mask_delta[:cnt] = 1
            # EOS
            label_index[self.seq_len-1] = 28 * 28
            label_onehot[self.seq_len-1, 28 * 28] = 1

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
    
    def rgb_img_read(self, img_path):
        """
        Read image and always return it as a RGB image (3D vector with 3 channels).
        """
        img = imread(img_path)
        if len(img.shape) == 2:
            img = color.gray2rgb(img)

        # Deal with RGBA
        img = img[..., :3]

        if img.dtype == 'uint8':
            # [0,1] image
            img = img.astype(np.float32)/255

        return img

    def extract_crop(self, data_info, context_expansion, rescale_imgsize=224):
        img = self.rgb_img_read(os.path.join(self.img_path, data_info['file_name']))
        
        # 多边形的顶点列表
        poly = np.array(data_info['polygon'][0], dtype=np.float).reshape(-1,2)  # (point_num, 2)
        xs = poly[:,0]
        ys = poly[:,1]

        bbox = data_info['bbox']
        x0, y0, w, h = bbox
        x_center = x0 + (1+w)/2.
        y_center = y0 + (1+h)/2.

        widescreen = True if w > h else False
        if not widescreen:
            img = img.transpose((1, 0, 2))
            x_center, y_center, w, h = y_center, x_center, h, w
            xs, ys = ys, xs

        x_min = int(np.floor(x_center - w*(1 + context_expansion)/2.))
        x_max = int(np.ceil(x_center + w*(1 + context_expansion)/2.))
        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(rescale_imgsize)/patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = skimgtransform.rescale(new_img, scale_factor, order=1, 
            preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)
        #assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

        xs = (xs - x_min) / float(patch_w)
        ys = (ys - (y_min-top_margin)) / float(patch_w)
        xs = np.clip(xs, 0 + EPS, 1 - EPS)
        ys = np.clip(ys, 0 + EPS, 1 - EPS)

        starting_point = [x_min, y_min-top_margin]
        if not widescreen:
            # Now that everything is in a square
            # bring things back to original mode
            new_img = new_img.transpose((1,0,2))
            starting_point = [y_min-top_margin, x_min]
            xs, ys = ys, xs

        poly = np.array([xs, ys]).T

        return_dict = {
            'img': new_img,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen,
            'poly': poly*rescale_imgsize
        }
        
        # for debug
        # import cv2
        # cv2.imwrite('test.png',(new_img*255).astype(np.int32))
        # timg=cv2.imread('test.png')
        # seg = (poly*rescale_imgsize).astype(np.int32)
        # dst = cv2.drawContours(timg, [seg], -1, (0,0,255), -1)
        # cv2.imwrite('dst.png',dst)

        return return_dict


"""
load data
"""
def loadData(img_path, anno_path, data_num, len_s, mode, batch_size, shuffle=True):
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform = transforms.Compose([transforms.ToTensor(), ])
    # pdb.set_trace()
    building = Building(data_num, len_s, mode, img_path, anno_path, transform)
    # building.__getitem__(0)
    dataloader = DataLoader(building, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False)

    print('DataLoader complete!', dataloader)
    return dataloader

if __name__ == '__main__':
    #import pdb;pdb.set_trace()
    img_path = '/home/zhangmingming_2020/data/building/building_coco/train/images'
    anno_file = '/home/zhangmingming_2020/data/building/building_coco/annotation/train.json'
    loader = loadData(img_path, anno_file, 16, 71, 'train', 2)
    
    print(len(loader))
