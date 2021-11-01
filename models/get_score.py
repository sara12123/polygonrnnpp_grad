# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.optim as optim
from models.polygon_model import PolygonModel
from utils import *
from dataloader import loadData
from losses import delta_loss
import warnings
import torch.nn as nn
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')

