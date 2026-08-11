from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import re
import pandas as pd
from normalization import get_local_bases,normalize_skeletons,get_start_poses,denormalize_skeletons
from attribute_computation import Attribute_Computation
import os

"""
get the default options
"""
class GaitPatternDatasetOptions:
    def __init__(self):
        self.base_dir= '/datasets_nas/human_activity/gaitpatterns/dataset/'      
        self.motion_dir=pjoin(self.base_dir,'joints_unscaled')
        self.label_dir=""#pjoin(self.base_dir,'labels/')
        self.computed_label_dir=pjoin(self.base_dir,'computed_labels_w30/') #default use the computed labels
        self.window_size = 30
        self.trainsplit=pjoin(self.base_dir,'train.txt')
        self.valsplit=pjoin(self.base_dir,'val.txt')
        self.testsplit=pjoin(self.base_dir,'test.txt')
        self.device="cpu"
        self.compute_labels=False #True
        self.save_labels="" # False

"""
Dataset class for providing gaitpatterns dataset
"""
class GaitPatternsDataset(data.Dataset):
    def __init__(self, opt, split_file):
        """
        create dataset from options
        opt.motion_dir - folder containing .npy files of the raw skeleton sequences
        opt.label_dir - folder with the associated labels
        opt.window_size - frame length of the sequences
        """
        if opt.device is None:
            self.device="cpu"
        else:
            self.device=opt.device
        self.opt = opt
        self.data = []
        self.data_normalized = []
        self.label = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        atcomp=Attribute_Computation()
        if hasattr(self.opt, 'save_labels') and self.opt.save_labels !="":
            if not os.path.exists(self.opt.save_labels):
                os.makedirs(self.opt.save_labels)

        for name in tqdm(id_list):
            #print("next file ####################################")
            try:
                #motion = pd.read_csv(pjoin(opt.motion_dir, name + '.csv'),  sep=';', header=None).values
                #motion = motion.reshape((motion.shape[0],int(motion.shape[1]/3),3))
                #print("motion shape ",motion.shape)
                #print("motion sample ",motion[0])
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))

                if motion.shape[0] < opt.window_size:
                    continue
                motion_tensor = torch.tensor(motion,device=self.device)
                self.data.append(motion_tensor)
                
                self.lengths.append(motion.shape[0] - opt.window_size)
                #use either the computed labels or the raw labels depending on the arguments set
                if (hasattr(opt, 'computed_label_dir')) and (opt.computed_label_dir!=""):
                    label= np.load(pjoin(opt.computed_label_dir, name + '.npy'))
                    label= torch.tensor(label)
                else:
                    if (not hasattr(opt, 'compute_labels')) or (opt.compute_labels != True):
                        label = pd.read_csv(pjoin(opt.label_dir, name + '.csv'),  sep=';', header=None).values
                        label = torch.tensor(label).repeat(motion_tensor.shape[0],1)
                        #print("label.shape",label.shape)
                
                if (hasattr(opt, 'compute_labels')) and (opt.compute_labels == True):
                    attr=atcomp.get_attributes(motion_tensor[0:opt.window_size,:,:])
                    label =torch.empty( (0,attr.shape[0]), dtype=torch.float32,device=self.device) # must adapt the size to the actual number of attributes provided
                    for i in range(0,motion_tensor.shape[0] - opt.window_size):
                         window_labels=atcomp.get_attributes(motion_tensor[i:i+ opt.window_size,:,:])
                         label=torch.cat((label,window_labels[None,:]),dim=0)
                    
                self.label.append(label)
                
                #store the normalized data as well
                normalized = normalize_skeletons(motion_tensor[None,:,:,:])
                #print(normalized.shape)
                self.data_normalized.append(normalized[0])

            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass
            if (hasattr(opt, 'save_labels')) and (opt.save_labels !=""):
                print("saved computed labels ", len(self.label))
                filename=pjoin(opt.save_labels, name + '.npy')
                np.save(filename,self.label[-1].numpy())

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        #print("label shape ",self.label[motion_id])
        label = self.label[motion_id][idx]
        motion_normalized = self.data_normalized[motion_id][idx:idx + self.opt.window_size]
        return motion, motion_normalized, label

