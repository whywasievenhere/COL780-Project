#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:44:26 2021

@author: shreyansnagori
"""


import pickle
import os
import numpy as np

file = open("partitions.pkl","rb")
dict1 = pickle.load(file)
# =============================================================================
print(dict1.keys())
#print(len(dict1['trainval_im_names']))
#print(dict1['train_im_names'])
#print(len(dict1['train_ids2labels']))
#print(len(dict1['val_im_names']))
#print(len(dict1['test_im_names']))
#print(len(dict1['test_marks'])) 
#print(dict1['trainval_ids2labels'])
# =============================================================================
# =============================================================================
list1= []

root= ""
for path, subdirs, files in os.walk("/home/s7rthak/COL780-Project/reid-col780-master/reid-col780-master/data/train"):
     for name in files:
         list1.append(os.path.join(path, name))
             
list2 = []
fin_list = []
for path, subdirs, files in os.walk("/home/s7rthak/COL780-Project/reid-col780-master/reid-col780-master/data/val/query"):
    for name in files:
        list2.append(os.path.join(path, name))
        fin_list.append(0)
for path, subdirs, files in os.walk("/home/s7rthak/COL780-Project/reid-col780-master/reid-col780-master/data/val/gallery"):
    for name in files:
        if name == ".DS_Store":
            continue
        list2.append(os.path.join(path, name))
        fin_list.append(1)
     
fin_list = np.array(fin_list)
print(len(list2))
dict2 = {}

def ids2label(list):
    train_files = set()
    for img_file in list1:
        train_files.add(int(img_file.split('/')[-2]))
    id2label = dict()
    cnt = 0
    for img_id in train_files:
        id2label[img_id] = cnt
        cnt += 1
    return id2label

list_tv = list1+list2
# list_tv = sorted(list_tv)
dict2['trainval_im_names']= list1
dict2['trainval_ids2labels']= ids2label(list1)
dict2['train_im_names']= list1


dict2['train_ids2labels']= ids2label(list1)
dict2['val_im_names']= list2
dict2['val_marks']= fin_list
dict2['test_im_names']= list2
dict2['test_marks']= fin_list
 
with open('/home/s7rthak/COL780-Project/reid-col780-master/reid-col780-master/partitions.pkl', 'wb') as handle:
    pickle.dump(dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     
# =============================================================================

