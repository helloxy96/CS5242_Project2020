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

from utils.data import get_file_split, get_max_prob_seg_seq


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
batch_size = 20

def evaluation(model, val_dataloader, type='segment'):

    predict_labels = []
    top_5_predict_labels = []
    groundTruth_labels = []
    outputs = []

    predict_scores = []
    top_5_scores = []

    model.eval()
    for x, labels in val_dataloader:
        x, labels = x.to(device), labels.to(device)

        with torch.no_grad():
            output = model(x).to(device)
            output = output.view(-1, 48)
            predict_label = torch.max(output, 1)[1]
            labels = labels.flatten()
            top_5_predict_label = torch.topk(output, k=5, dim=1)[1]
            outputs.extend(output.tolist())

        top_5_scores.append(torch.sum(top_5_predict_label == labels.unsqueeze(dim=1)).item())
        top_5_predict_labels.extend(top_5_predict_label.tolist())


        labels = labels.cpu().data.squeeze().numpy()
        predict_label = predict_label.long().cpu().data.squeeze().numpy()

        # frame check
        # print(labels, predict_label, labels)
        
        if labels.size > 1:
            #step_score = accuracy_score(labels, predict_label)
            step_score = np.sum(labels == predict_label)
        else:
            step_score = 1 if labels == predict_label else 0
            labels = np.array([labels])
            predict_label = np.array([predict_label])
            # print(predict_label, labels)

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
        seg_acc = sum(predict_scores) / len(predict_labels)
        top_5_seg_scc = sum(top_5_scores) / len(predict_labels)

        # result
        res = {
            'seg': {
                'acc': seg_acc,
                'predict': predict_labels,
                'groundTruth': groundTruth_labels,
                'output': outputs
            }
        }

        print('valid seg acc:', seg_acc, '\n')
        print('valid top_5_seg_scc:', top_5_seg_scc, '\n')

    return res

def get_each_file_acc(predict, groundTruth, file_splits, split_load):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']

    # get each file acc
    s_index = 0
    print(len(file_splits), len(content_all))
    for i, num in enumerate(file_splits):
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


def eval_with_prob(outputs, groundTruths, predicted, prob_mat, h0_vec, file_splits):
    s_index = 0
    new_outputs = []
    new_labels = []

    outputs = torch.tensor(outputs).to(device)

    correct = 0
    for f_split in file_splits:
        f_outputs = outputs[s_index:s_index+f_split]
        f_gt = groundTruths[s_index:s_index+f_split]
        f_oldlabel = predicted[s_index:s_index+f_split]

        new_label = get_max_prob_seg_seq(f_outputs, prob_mat, h0_vec) 

        if new_label != f_oldlabel:
            print('\nold labels', f_oldlabel)
            print('new labels', new_label)
            print('groundTruth', f_gt)

        correct += torch.sum(
            torch.tensor(new_label) == torch.tensor(f_gt)
        ).item()

        new_labels.extend(new_label)

        s_index += f_split
    
    print('seg acc:', correct / len(groundTruths) )

    res = {
        'predict': new_labels,
        'groundTruth': groundTruths 
    }
    return res

def get_each_label_acc(predict, groundTruth):
    labels_total = torch.zeros(48)
    labels_correct = torch.zeros(48)
    labels_wrong_dict = {}

    labels_wrong_dict[0] = {}
    for i, p in enumerate(predict):
        label = groundTruth[i]
        labels_total[label] +=1

        if p == label:
            labels_correct[label] += 1
        else:
            if label not in labels_wrong_dict:
                labels_wrong_dict[label] = {}
            if p not in labels_wrong_dict[label]:
                labels_wrong_dict[label][p] = 0
            labels_wrong_dict[label][p] += 1

    labels_acc = labels_correct / labels_total
    sorted_, indexs = torch.sort(labels_acc, descending=True)

    print('\nsorted acc')
    for i in range(len(indexs)):
        label = indexs[i].item()
        if label not in labels_wrong_dict:
            labels_wrong_dict[label] = {}
        print(f'\nlabel {label} acc: {sorted_[i]}')
        print(f'groundTruth {label} but predict as: \n', labels_wrong_dict[label])
        


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

    prob_mat = torch.load('./trained/conv/prob_mat_next.pt', map_location=torch.device(device))
    h0_vec = torch.load('./trained/conv/prob_h0.pt', map_location=torch.device(device))

    print(prob_mat)

    file_splits = get_file_split(val_split, GT_folder, actions_dict)

    res_with_probmat = eval_with_prob(
        res['seg']['output'],
        res['seg']['groundTruth'],
        res['seg']['predict'],
        prob_mat,
        h0_vec,
        file_splits
    )

    print('------original file predict result--------\n')

    get_each_file_acc(
        res['seg']['predict'],
        res['seg']['groundTruth'],
        file_splits,
        val_split
    )

    print('------with prob, file predict result--------\n')

    get_each_file_acc(
        res_with_probmat['predict'],
        res_with_probmat['groundTruth'],
        file_splits,
        val_split
    )

    print('------original label predict result--------\n')

    get_each_label_acc(
        res['seg']['predict'],
        res['seg']['groundTruth']
    )

    print('------with prob, label predict result--------\n')

    get_each_label_acc(
        res_with_probmat['predict'],
        res_with_probmat['groundTruth']
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
