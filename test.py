from __future__ import print_function

import argparse
import itertools
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# print(sys.path)
print("Is cuda available?")
print(torch.cuda.is_available())

t = torch.cuda.get_device_properties(0).total_memory
c = torch.cuda.memory_cached(0)
a = torch.cuda.memory_allocated(0)
f = c-a  # free inside cache

print("total memory")
print(t)
print("cached memory")
print(c)
print("allocated memory")
print(a)
print("free memory")
print(f)

import code.archs as archs
from code.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from code.utils.cluster.transforms import sobel_process
from code.utils.segmentation.segmentation_eval import \
  segmentation_eval
from code.utils.segmentation.IID_losses import IID_segmentation_loss, \
  IID_segmentation_loss_uncollapsed
from code.utils.segmentation.data import segmentation_create_dataloaders
from code.utils.segmentation.general import set_segmentation_input_channels

