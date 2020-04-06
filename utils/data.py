import torch
from .read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np

def get_file_split_by_segtxt(seg_info_path):
    splits = []        
    f = open(seg_info_path, 'r')

    for f_i, line in enumerate(f):

        indexes = line.split()

        splits.append(len(indexes)-1)

    return splits

def get_file_split(bundle_path, GT_folder, actions_dict):
    file_ptr = open(bundle_path, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']

    print(bundle_path, GT_folder)

    seg_nums_in_all_files = []
    for content in content_all:
        file_ptr = open(GT_folder + content, 'r')
        curr_gt = file_ptr.read().split('\n')[:-1]
        label_curr_video = []

        for iik in range(len(curr_gt)):
            label_curr_video.append(actions_dict[curr_gt[iik]])
        
        seg_num_in_a_file = 0
        for i in range(1, len(label_curr_video)):
            label = label_curr_video[i]
            last_label = label_curr_video[i-1]

            if last_label != label:
                seg_num_in_a_file += 1
        seg_nums_in_all_files.append(seg_num_in_a_file-1)

    return seg_nums_in_all_files

def get_class_prob_matrix(files_labels, file_splits):
    count_mat = torch.ones(48, 48)
    total_mat = torch.zeros(48)
    total_mat.fill_(48)
    
    s_index = 0
    for f_splits in file_splits:
        f_labels = files_labels[s_index:s_index+f_splits]

        for i, f_label in enumerate(f_labels):
            total_mat[f_label] += 1
            for j, other_f_label in enumerate(f_labels):
                if i != j:
                    count_mat[f_label][other_f_label] += 1

        s_index += f_splits

    prob_mat = count_mat / total_mat.unsqueeze(dim=0).permute(1, 0)

    return prob_mat

if __name__ == '__main__':
    COMP_PATH = '../'

    ''' 
    training to load train set
    test to load test set
    '''
    split = 'training'
    #split = 'test'
    train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle') #Train Split
    # train_split =  os.path.join(COMP_PATH, 'splits/dev_train.split1.bundle') #Train Split
    val_split   =  os.path.join(COMP_PATH, 'splits/val.split1.bundle')
    test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
    GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
    DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
    mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

    actions_dict = read_mapping_dict(mapping_loc)

    data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #
    file_splits = get_file_split(train_split, GT_folder, actions_dict)

    prob_mat = get_class_prob_matrix(data_labels, file_splits)

    print(prob_mat)
    torch.save(prob_mat, '../trained/conv/prob_mat.pt')