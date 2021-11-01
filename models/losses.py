# -*- coding: utf-8 -*-
from __future__ import print_function
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
from scipy.ndimage.morphology import distance_transform_cdt
import warnings
warnings.filterwarnings("ignore")

def delta_loss(pred_logits, gt, mask):
    """
    delta loss
    :param pred_logits: (bs*seq_len, 15*15)
    :param gt: (bs, 14*14)
    :param mask: (-1)
    :return:
    """
    bs = pred_logits.shape[0]

    loss = torch.sum(-gt * torch.nn.functional.log_softmax(pred_logits, dim=-1), dim=1)

    loss = (loss * mask.type_as(loss)).view(bs, -1)

    loss = torch.sum(loss, dim=1)

    real_pointnum = torch.sum(mask.contiguous().view(bs, -1), dim=1)
    loss = loss / real_pointnum
    return torch.mean(loss)


def iou_loss(poly1, poly2, mask):
    pass