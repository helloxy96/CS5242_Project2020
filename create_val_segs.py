from read_datasetBreakfast import read_mapping_dict
import os
import numpy as np
import random

COMP_PATH = ''

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
    chosen_seg_lists = random_chosen_segs(labels_segments_dict, n_seg_per_label, total_val_seg_num)

    # write
    file_ptr = open(train_split, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_select_val = [content_all[idx] for idx in range(len(content_all)) if idx in chosen_seg_lists]
    print(content_select_val)

    val_file_path = os.path.join(COMP_PATH, 'splits/val.split1.bundle')
    f = open(val_file_path, 'w+')
    for seg_path in content_select_val:
        f.write(seg_path)
        f.write('\n')
    f.close()