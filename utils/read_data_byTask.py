# -*- coding: utf-8 -*-
import os  
import torch
import numpy as np
import os.path
import re
 
 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

 
def load_data(split_load, actions_dict, GT_folder, DATA_folder, datatype = 'training'):
    """
    return: data_breakfast, dict, keys are tasks
            labels_breakfast, dict, only for training, keys are tasks
    """
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']
    
    task_all = [re.findall('_([a-z]*?).txt', t)[0] for t in content_all]
    
    if datatype == 'training':
        data_breakfast = {task:[] for task in task_all}
        labels_breakfast = {task:[] for task in task_all}
        
        for content in content_all:

            task = re.findall('_([a-z]*?).txt', content)[0]    # task: cereals
            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])

            data_breakfast[task].append(curr_data)
            labels_breakfast[task].append(label_curr_video)

        return data_breakfast, labels_breakfast
        
    if datatype == 'test':
        data_breakfast = {task:[] for task in task_all}
        
        segment = []
        for content in content_all:

            task = re.findall('_([a-z]*?).txt', content)[0]    # task: cereals
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
                                 
            data_breakfast[task].append(curr_data)
        
        return data_breakfast

def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict=dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

if __name__ == "__main__":
    COMP_PATH = '../'
    split = 'training'
    #split = 'test'
    train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/')
    DATA_folder =  os.path.join(COMP_PATH, 'data/')
    mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')
    
  
    actions_dict = read_mapping_dict(mapping_loc)
    if  split == 'training':
        data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split)
        
    if  split == 'test':
        data_feat = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split)