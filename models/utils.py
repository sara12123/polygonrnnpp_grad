# -*- coding: utf-8 -*-
from __future__ import print_function
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
from scipy.ndimage.morphology import distance_transform_cdt
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    cv2.polylines(mask, [poly], True, [1])

    return mask


def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask


def iou(vertices1, vertices2, h ,w):
    '''
    calculate iou of two polygons
    :param vertices1: vertices of the first polygon
    :param vertices2: vertices of the second polygon
    :return: the iou, the intersection area, the union area
    '''
    img1 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)  # fill=1表示polygon内部填充为1
    mask1 = np.array(img1)
    img2 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img2).polygon(vertices2, outline=1, fill=1)
    mask2 = np.array(img2)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    nu = np.sum(intersection)
    de = np.sum(union)
    if de!=0:
        return nu*1.0/de, nu, de
    else:
        return 0, nu, de


def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou


def iou_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred:
        masks[0] = draw_poly(masks[0], p)

    for g in gt:
        masks[1] = draw_poly(masks[1], g)

    return iou_from_mask(masks[0], masks[1]), masks


def label2vertex(labels):
    '''
    convert 1D labels to 2D vertices coordinates
    :param labels: 1D labels
    :return: 2D vertices coordinates: [(x1, y1),(x2,y2),...]
    '''
    vertices = []
    for label in labels:
        if (label == 784):
            break
        vertex = ((label % 28) * 8, (label / 28) * 8)
        vertices.append(vertex)
    return vertices


def getbboxfromkps(kps,h,w):
    '''

    :param kps:
    :return:
    '''
    min_c = np.min(np.array(kps), axis=0)
    max_c = np.max(np.array(kps), axis=0)
    object_h = max_c[1] - min_c[1]
    object_w = max_c[0] - min_c[0]
    h_extend = int(round(0.1 * object_h))
    w_extend = int(round(0.1 * object_w))
    min_row = np.maximum(0, min_c[1] - h_extend)
    min_col = np.maximum(0, min_c[0] - w_extend)
    max_row = np.minimum(h, max_c[1] + h_extend)
    max_col = np.minimum(w, max_c[0] + w_extend)
    return (min_row,min_col,max_row,max_col)


def img2tensor(img):
    '''

    :param img:
    :return:
    '''
    img = np.rollaxis(img,2,0)
    return torch.from_numpy(img)


def tensor2img(tensor):
    '''

    :param tensor:
    :return:
    '''
    img = (tensor.numpy()*255).astype('uint8')
    img = np.rollaxis(img,0,3)
    return img


# smooth RNN target
def dt_targets_from_class(poly, grid_size, dt_threshold):
    #  grid_size: 28
    #  poly: 即Ground Truth, [bs, seq_len], 每行一个poly, seq_len个值, 每个值是在grid*gird中的index

    """
    NOTE: numpy function!
    poly: [bs, time_steps], each value in [0, grid*size**2+1)
    grid_size: size of the grid the polygon is in
    dt_threshold: threshold for smoothing in dt targets

    returns:
    full_targets: [bs, time_steps, grid_size**2+1] array containing
    dt smoothed targets to be used for the polygon loss function
    """
    full_targets = []
    for b in range(poly.shape[0]):
        targets = []
        for p in poly[b]:
            t = np.zeros(grid_size**2+1, dtype=np.int32)
            t[p] += 1

            if p != grid_size**2:#EOS
                spatial_part = t[:-1]
                spatial_part = np.reshape(spatial_part, [grid_size, grid_size, 1])

                # Invert image
                spatial_part = -1 * (spatial_part - 1)
                # Compute distance transform
                spatial_part = distance_transform_cdt(spatial_part, metric='taxicab').astype(np.float32)
                # Threshold
                spatial_part = np.clip(spatial_part, 0, dt_threshold)
                # Normalize
                spatial_part /= dt_threshold
                # Invert back
                spatial_part = -1. * (spatial_part - 1.)

                spatial_part /= np.sum(spatial_part)
                spatial_part = spatial_part.flatten()

                t = np.concatenate([spatial_part, [0.]], axis=-1)

            targets.append(t.astype(np.float32))
        full_targets.append(targets)

    return np.array(full_targets, dtype=np.float32)


def reverse(arr, start, end):
    while start < end:
        temp = arr[start]
        arr[start] = arr[end]
        arr[end] = temp
        start += 1
        end -= 1


# numpy算法，arr, 向右循环启动k位
def rightShift(arr, k):
    if arr is None:
        print("参数不合法！")
        return
    lens = len(arr)
    k %= lens
    reverse(arr, 0, lens - k - 1)
    reverse(arr, lens - k, lens - 1)
    reverse(arr, 0, lens - 1)


def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]   # dp 道格拉斯~算法，多边形点曲线拟合， 并减少点的数量

    return poly


def poly0g_to_poly01(polygon, grid_side):
    """
    [0, grid_side] coordinates to [0, 1].
    Note: we add 0.5 to the vertices so that the points
    lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5)/grid_side

    return result


def class_to_grid(poly, out_tensor, grid_size):
    """
    NOTE: Torch function
    accepts out_tensor to do it inplace

    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    # Remove old state of out_tensor

    b = 0
    for i in poly:
        if i < grid_size * grid_size:
            x = (i%grid_size).long()
            y = (i/grid_size).long()
            out_tensor[b,0,y,x] = 1
        b += 1

    return out_tensor


def get_masked_poly(poly, grid_size):
    """
    NOTE: Numpy function

    Given a polygon of shape (N,), finds the first EOS token
    and masks the predicted polygon till that point
    """
    if np.max(poly) == grid_size**2:
        # If there is an EOS in the prediction
        length = np.argmax(poly)
        poly = poly[:length]
        # This automatically removes the EOS

    return poly


def class_to_xy(poly, grid_size):
    """
    NOTE: Numpy function
    poly: [bs, time_steps] or [time_steps]

    Returns: [bs, time_steps, 2] or [time_steps, 2]
    """
    x = (poly % grid_size).astype(np.int32)
    y = (poly / grid_size).astype(np.int32)

    out_poly = np.stack([x,y], axis=-1)

    return out_poly


def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    mask[poly[:, 1], poly[:, 0]] = 1.

    return mask


def prepare_delta_target(rnn_pred, polygon_gt):
    """
    Generate the target of delta
    :param rnn_pred: [bs, len_s]
    :param polygon_gt: [bs, len_s, 2]
    :return:
    """

    rnn_pred = rnn_pred[:, :-1]  # -eos
    bs = rnn_pred.shape[0]
    l = rnn_pred.shape[1]
    delta = torch.zeros((bs, l)).cuda()  # 在14*14 grid中的坐标->index

    # pred = rnn_pred[:, :-1]  # [bs, len_s]
    # # To(0, 112)
    # gt = (polygon_gt + 0.5) // 2  # ~112 下整除了  (bs, 71, 2)
    # To(0, 112)
    # pred_poly_x = (pred % 28) * 4.0 + 2  # (bs, 71)
    # pred_poly_y = (pred // 28) * 4.0 + 2
    # delta_x = gt[:, :, 0] - pred_poly_x + 7  # (bs, 71)
    # delta_y = gt[:, :, 1] - pred_poly_y + 7  # (bs, 71)
    # index1 = (delta_x > 14)
    # delta_x[index1] = 14
    # index2 = (delta_x < 0)
    # delta_x[index2] = 0
    # index1 = (delta_y > 14)
    # delta_y[index1] = 14
    # index2 = (delta_y < 0)
    # delta_y[index2] = 0
    # delta = delta_x + 15 * delta_y  # (bs, 71)

    for b in range(bs):
        pred = rnn_pred[b]
        gt = (polygon_gt[b] + 0.5) // 2  # ~112 下整除了
        pred_poly_x = (pred % 28) * 4.0 + 2
        pred_poly_y = (pred // 28) * 4.0 + 2
        delta_x = gt[:, 0] - pred_poly_x + 7  # (71)
        delta_y = gt[:, 1] - pred_poly_y + 7
        index1 = (delta_x > 14)
        delta_x[index1] = 14
        index2 = (delta_x < 0)
        delta_x[index2] = 0
        index1 = (delta_y > 14)
        delta_y[index1] = 14
        index2 = (delta_y < 0)
        delta_y[index2] = 0
        delta[b] = delta_x + 15 * delta_y
    return delta.long()


"""
local_x = 7 + x
if local_x >= 0:
    local_x = min(14, local_x)
if local_x < 0:
    local_x = 0
local_y = 7 + y
if local_y >= 0:
    local_y = min(14, local_y)
if local_y < 0:
    local_y = 0
local_prediction.append(local_x + 15 * local_y)
"""