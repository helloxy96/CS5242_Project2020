import torch
from .read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch.nn.functional as F

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
    count_mat = torch.zeros(48, 48)
    total_mat = torch.zeros(48)
    # total_mat.fill_(48)
    
    h0_vec = torch.zeros(48)

    s_index = 0
    segs_total = 0
    for f_splits in file_splits:
        f_labels = files_labels[s_index:s_index+f_splits]

        # start label put into h0_vec
        if len(f_labels) > 0:
            h0_vec[f_labels[0]] += 1

        for i in range(len(f_labels)-1):

            f_label = f_labels[i]
            next_label = f_labels[i+1]

            total_mat[f_label] += 1
            count_mat[f_label][next_label] += 1

            segs_total += 1

        s_index += f_splits

    prob_mat = count_mat / total_mat.unsqueeze(dim=0).permute(1, 0)

    mask = (torch.isnan(prob_mat)) | (prob_mat == 0)
    avg_prob = torch.mean(prob_mat[~mask])
    prob_mat[mask] = avg_prob / 3

    h0_vec = h0_vec / segs_total
    h0_mask = h0_vec == 0
    h0_vec[h0_mask] = torch.mean(h0_vec[~h0_mask]) / 3

    return prob_mat, h0_vec

def normalize(x,dim=1):
    # maxmin normalize
    _min = x.min(dim, keepdim=True)[0]
    _max = x.max(dim, keepdim=True)[0]

    res = (x-_min) / (_max-_min)

    if len(_min.shape) == 2:
        mask = (_min == _max).squeeze(dim=1)
        res[mask] = 1.0 / res.shape[1]
    else:
        if _min == _max:
            res[:] = 1.0 / res.shape[0]

    return res

def get_max_prob_seg_seq(outputs, prob_mat, h0_vec):
    device = outputs.device
    # 48
    action_nums = prob_mat.shape[0]
    f = torch.zeros(len(outputs), action_nums).to(device)
    record = torch.zeros(len(outputs), action_nums).to(device).long()
    norm_prob_mat = normalize(prob_mat, dim=1)

    for i, i_prob in enumerate(outputs):
        norm_i_prob = normalize(i_prob, dim=0)

        # do softmax on output
        i_prob = torch.log(F.softmax(i_prob) + 1e-4)


        if i == 0:
            f[i] = h0_vec + i_prob
        else:
            for j in range(action_nums):
                last_to_j_max_prob, last_j = torch.max(f[i-1] + torch.log(prob_mat[:, j]), 0)
                f[i][j] = last_to_j_max_prob + i_prob[j]
                # print(last_to_j_max_prob, i_prob[j])
                record[i][j] = last_j
        # f[i] = normalize(f[i],dim=0)

    # get final max prob
    max_final_prob, max_final_action = torch.max(f[len(outputs)-1], 0)

    # get max prob seq
    seq = [max_final_action.item()]
    for k in reversed(range(1, len(outputs))):
        last_action = record[k][max_final_action]
        seq.append(last_action.item())
        max_final_action = last_action
    
    seq.reverse()
    return seq

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

    prob_mat, h0_vec = get_class_prob_matrix(data_labels, file_splits)

    print(prob_mat)
    torch.save(prob_mat, '../trained/conv/prob_transform_mat.pt')
    print(h0_vec)
    torch.save(h0_vec, '../trained/conv/prob_h0.pt')
