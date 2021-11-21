import os
import csv
import torch
from collections import OrderedDict
import os
import os.path as osp
import cPickle as pickle
from scipy import io
import datetime
import time
from contextlib import contextmanager
from torch.autograd import Variable

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        label = path.split("/")[-2]
        filename = os.path.basename(path)
        camera = filename.split('_')[0]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

