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

    :param split_load:
    :param actions_dict:
    :param GT_folder:
    :param DATA_folder:
    :param datatype:
    :return: data_breakfast, dict, keys are tasks  [video[segment[frame]]]
                {'cereals':[[[array(...)], [array(...)]]]}
            labels_breakfast, dict, only for training, keys are tasks   [video[segment[frame]]]
                {'cereals':[[[0,0,2,4...], [0,0,5,5....]]}
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

            label_seq, length_seq = get_label_length_seq(label_curr_video)
            data_by_seg = []
            label_by_seg = []
            for seg_idx in range(len(length_seq)-1):
                data_by_seg.append(curr_data[length_seq[seg_idx]:length_seq[seg_idx+1]])
                label_by_seg.append(label_curr_video[length_seq[seg_idx]:length_seq[seg_idx+1]])
                
            data_breakfast[task].append(data_by_seg)
            labels_breakfast[task].append(label_by_seg)

        return data_breakfast, labels_breakfast
        
    if datatype == 'test':
        data_breakfast = {task:[] for task in task_all}
        
        test_seg_list = get_test_seq()
        for idx, content in enumerate(content_all):

            task = re.findall('_([a-z]*?).txt', content)[0]    # task: cereals
            seg_list = test_seg_list[idx]

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            
            data_by_seg = []
            for seg_idx in range(len(seg_list)+1):
                if seg_idx == 0:
                    data_by_seg.append(curr_data[:seg_list[seg_idx]])
                elif seg_idx == len(seg_list):
                    data_by_seg.append(curr_data[seg_list[seg_idx-1]:])
                else:
                    data_by_seg.append(curr_data[seg_list[seg_idx-1]:seg_list[seg_idx]])
                                 
            data_breakfast[task].append(data_by_seg)
            
        return data_breakfast



def get_test_seq():
    f = open('../test_segment.txt', 'r+', newline='\n')
    all_segs = f.read().splitlines()
    test_seg_list = []
    for seg in all_segs:
        seg_list = seg.split()
        seg_list = [int(s) for s in seg_list]
        test_seg_list.append(seg_list)
    return test_seg_list


def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))
    
    if content[-1] != 0:
        label_seq.append(content[-1])
    
    return label_seq, length_seq


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