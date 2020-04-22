#!/usr/bin/env python
# coding: utf-8

# In[2]:


from read_datasetBreakfast import read_mapping_dict
import os
import numpy as np
import random

COMP_PATH = '../'

''' 
training to load train set
test to load test set
'''
split = 'training'


train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle')  # Train Split
GT_folder = os.path.join(COMP_PATH, 'groundTruth/')  # Ground Truth Labels for each training video
mapping_loc = os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

actions_dict = read_mapping_dict(mapping_loc)

# copy and revise from read_datasetBreakfast.py
def load_data_label(split_load, actions_dict, GT_folder, datatype='training', ):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    if datatype == 'training':
        labels_breakfast = []
        for content in content_all:

            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]

            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])

            labels_breakfast.append(label_curr_video)

        labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
        print("Finish Load the Training data and labels!!!")
        return labels_uniq


def get_label_bounds(data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(0, len(data_labels)):
        uniq_group, indc_group = get_label_length_seq(data_labels[kki])
        labels_uniq.append(uniq_group[1:-1])
        labels_uniq_loc.append(indc_group[1:-1])
    return labels_uniq, labels_uniq_loc


def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            # print(content[i])
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    if content[-1] != 0:
        label_seq.append(content[-1])

    return label_seq, length_seq


def get_labels_segs_dict(data_labels):
    '''
    get the segs list that contain each label
    :param data_labels:
    :return: {label1:[seg_idx1, seg_idx2,...], label2:[...]}
    '''
    labels_segments_dict = {}
    for i in range(len(data_labels)):
        seg_labels = data_labels[i]
        for label in seg_labels:
            if label in labels_segments_dict:
                if i not in labels_segments_dict[label]:
                    labels_segments_dict[label].append(i)
            else:
                labels_segments_dict[label] = [i]
    return labels_segments_dict


def random_chosen_segs(labels_segments_dict, n_seg_per_label, total_val_seg_num):
    """
    random choose 4-5 segments for each label, totally 200 segments for all labels
    :param labels_segments_dict:
    :param n_seg_per_label:
    :param total_val_seg_num:
    :return:
    """
    chosen_seg_lists = []
    random.seed(1)
    while len(chosen_seg_lists) < total_val_seg_num:
        for label, seg_idx in labels_segments_dict.items():
            if len(seg_idx) <= n_seg_per_label:
                chosen_seg_lists.extend(seg_idx)
            else:
                chosen_seg_lists.extend(random.sample(seg_idx, n_seg_per_label))
        n_seg_per_label += 1

        chosen_seg_lists = list(set(chosen_seg_lists))
    chosen_seg_lists.sort()
    return chosen_seg_lists


if __name__ == '__main__':
    data_labels = load_data_label(train_split, actions_dict, GT_folder, datatype=split)
    labels_segments_dict = get_labels_segs_dict(data_labels)

    n_seg_per_label = 5     # segments num chosen for each label
    total_val_seg_num = 200         # total segments min num chosen for all labels
#     chosen_seg_lists = random_chosen_segs(labels_segments_dict, n_seg_per_label, total_val_seg_num)

#     # write
#     file_ptr = open(train_split, 'r')
#     content_all = file_ptr.read().split('\n')[1:-1]
#     content_select_val = [content_all[idx] for idx in range(len(content_all)) if idx in chosen_seg_lists]
#     print(content_select_val)
#     content_not_select_train = [content_all[idx] for idx in range(len(content_all)) if idx not in chosen_seg_lists]
#     print(content_not_select_train)

#     val_file_path = os.path.join(COMP_PATH, 'splits/val.split1.bundle')
#     f = open(val_file_path, 'w+')
#     f.write('#bundle\n')
#     for seg_path in content_select_val:
#         f.write(seg_path)
#         f.write('\n')
#     f.close()

#     val_file_path = os.path.join(COMP_PATH, 'splits/train.exclude_val.bundle')
#     f = open(val_file_path, 'w+')
#     f.write('#bundle\n')
#     for seg_path in content_not_select_train:
#         f.write(seg_path)
#         f.write('\n')
#     f.close()
# print(chosen_seg_lists[:3])
# print(data_labels[:3])


# ## detect problem

# In[28]:


def random_chosen_segs(data_labels, labels_segments_dict, limit_label, total_val_seg_num):
    """
    random choose 4-5 segments for each label, totally 200 segments for all labels
    :param labels_segments_dict:
    :param n_seg_per_label:
    :param total_val_seg_num:
    :return:
    """
    chosen_seg_lists = []
#     random.seed(1)
    split_idx_list = list(range(len(data_labels)))    # 0-1459共1460个splits
    counting_label_dict = {label:limit_label for label in list(labels_segments_dict.keys())}
    
    not_chosen_list = split_idx_list
    kk = 0
    pre_temp_list = []
    while len(chosen_seg_lists) <= total_val_seg_num and kk < 20:
        temp_chosen_splits_list = random.sample(not_chosen_list, 100)
        for idx in temp_chosen_splits_list:
            flag = 1
            split_labels = data_labels[idx]
            for l in split_labels:
                if counting_label_dict[l]>0:
                    pass
                else:
                    flag = 0
            if flag == 0:
                continue
            else:    # each label of the splits hasn't over the limit
                chosen_seg_lists.append(idx)
                for l in split_labels:
                    counting_label_dict[l] -= 1
        
        not_chosen_list = [i for i in split_idx_list if i not in chosen_seg_lists]
        pre_temp_list = temp_chosen_splits_list
        kk += 1
        print('\nround ', kk)
        print('first 3 of random select', temp_chosen_splits_list[:3])
        print('chosen num', len(chosen_seg_lists))
        print('not chosen num', len(not_chosen_list))

    
    return chosen_seg_lists

chosen_seg_lists = random_chosen_segs(data_labels, labels_segments_dict, 30, 200)

all_chosen_labels = []
for seg in chosen_seg_lists:
    all_chosen_labels.extend(data_labels[seg])
# print(all_chosen_labels)
count_dict = {}
for l in all_chosen_labels:
    if l not in count_dict:
        count_dict[l] = 1
    else:
        count_dict[l] += 1
print('\n chosen label count\n', count_dict)


# ### focusing on very small one

# In[33]:


# 选了只有3个选不出更多的 label 47，找到是包含47的所有splits打印出labels
# 发现 47总是和6，45，46 这样的高频词 label一起，别的几个限制了包含 47 的 splits入选
# print('segs contain label 47\n', labels_segments_dict[47])
# for i in labels_segments_dict[47]:
#     print(i, ' ', data_labels[i])

# print('\nsegs contain label 31\n', labels_segments_dict[31])
# for i in labels_segments_dict[31]:
#     print(i, ' ', data_labels[i])

labels = [16, 31, 41, 23, 38, 4, 47, 3, 8, 7, 40, 9, 18, 22, 30, 43]
for label in labels:
    print(f'\nsegs contain label {label}\n', labels_segments_dict[label])
    for i in labels_segments_dict[label]:
        print(i, ' ', data_labels[i])

