# eval 
from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch

from Dataset.VideoDataset import VideoDataset
from Dataset.SegmentDataset import SegmentDataset

import torch.utils.data as tud
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import sys

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
batch_size = 20

def evaluation(model, val_dataloader, type='segment'):

    predict_labels = []
    groundTruth_labels = []

    predict_scores = []

    model.eval()
    for x, labels in val_dataloader:
        x, labels = x.to(device), labels.to(device)

        with torch.no_grad():
            output = model(x).to(device)
            output = output.view(-1, 48)
            predict_label = torch.max(output, 1)[1]

        labels = labels.cpu().data.squeeze().numpy()
        predict_label = predict_label.long().cpu().data.squeeze().numpy()

        # frame check
        step_score = accuracy_score(labels, predict_label)
        # print('step score', step_score)
        predict_scores.append(step_score)
        
        predict_labels.extend(predict_label.tolist())
        groundTruth_labels.extend(labels.tolist())

    # if frame, evaluate frame and seg acc at the same time
    if type == 'frame':
        # get seg labels
        segs = []
        results = []
        for i, groundTruth in enumerate(groundTruth_labels):
            if i > 0 and groundTruth != groundTruth_labels[i-1]:
                segs.append(i)
                results.append(groundTruth)

        # segment check
        total = len(segs)
        correct_nums = 0

        predict_seg_labels = []
        groundTruth_seg_labels = []

        for i in range(total):
            seg_idx = segs[i]
            next_seg_idx = len(predict_labels) if i >= total-1 else segs[i+1]

            # print(seg_idx, next_seg_idx)
            seg_predict_label = get_most_frequent(predict_labels[seg_idx:next_seg_idx])

            if seg_predict_label == results[i]:
                correct_nums += 1
            
            predict_seg_labels.append(seg_predict_label)
            groundTruth_seg_labels.append(results[i])

        frame_acc_avg = sum(predict_scores) / len(predict_scores)
        seg_acc = correct_nums / total

        print('valid frame acc avg:', frame_acc_avg, '\n')
        print('valid seg acc:', seg_acc, '\n')

        # result
        res = {
            'frame': {
                'acc': frame_acc_avg,
                'predict': predict_labels,
                'groundTruth': groundTruth_labels
            },
            'seg': {
                'acc': seg_acc,
                'predict': predict_seg_labels,
                'groundTruth': groundTruth_seg_labels
            }
        }

    # if segment type, evalute seg acc directly
    elif type == 'segment':
        seg_acc = sum(predict_scores) / len(predict_scores)

        # result
        res = {
            'seg': {
                'acc': seg_acc,
                'predict': predict_labels,
                'groundTruth': groundTruth_labels
            }
        }

        print('valid seg acc:', seg_acc, '\n')

    return res

def get_each_file_acc(predict, groundTruth, split_load, actions_dict, GT_folder):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']

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

    # get each file acc
    s_index = 0
    print(len(seg_nums_in_all_files), len(content_all))
    for i, num in enumerate(seg_nums_in_all_files):
        f_predict = predict[s_index:s_index+num]
        f_groundTruth = groundTruth[s_index:s_index+num]

        # acc
        file_name = content_all[i]
        acc = accuracy_score(f_predict, f_groundTruth)
        print(f'acc of {file_name}: {acc}')
        print('groundTruth:', f_groundTruth)
        print('predict:', f_predict, '\n')

        s_index += num

def get_most_frequent(arr):
    arr = np.array(arr)
    return np.argmax(np.bincount(arr.astype(np.int64)))

def get_dataloader(type='segment'):
    
    valid_feat, valid_labels = load_data( val_split, actions_dict, GT_folder, DATA_folder, datatype = split, target=type) #

    if type == 'segment':
        valid_dataset = SegmentDataset(valid_feat, valid_labels)
    else:
        valid_dataset = VideoDataset(valid_feat, valid_labels)
    
    valid_dataloader = tud.DataLoader(valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers = 8, 
        pin_memory=True
    )
    return valid_dataloader


COMP_PATH = './'
split = 'training'

val_split   =  os.path.join(COMP_PATH, 'splits/val.split1.bundle')
mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')
GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos

actions_dict = read_mapping_dict(mapping_loc)



if __name__ == "__main__":

    trained_model_path = sys.argv[1]
    model_type = sys.argv[2] 

    model = torch.load(trained_model_path, map_location=torch.device(device))
    dataloader = get_dataloader(type=model_type)

    res = evaluation(model, dataloader)

    get_each_file_acc(
        res['seg']['predict'],
        res['seg']['groundTruth'],
        val_split,
        actions_dict,
        GT_folder,
    )

    seg_res = np.concatenate((
        np.array(res['seg']['predict']),
        np.array(res['seg']['groundTruth'])
    ))

    np.save('./results/conv/val_seg_out.npy', seg_res)

    if model_type == 'frame':
        frame_res = np.concatenate((
            np.array(res['frame']['predict']),
            np.array(res['frame']['groundTruth'])
        ), axis = 1)

        np.save('./results/conv/val_frame_out.npy', frame_res)
