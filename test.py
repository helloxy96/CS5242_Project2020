# eval 
from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os.path as path
import numpy as np
import torch

from Dataset.TestDataset import TestDataset

import torch.utils.data as tud
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
from utils.data import get_file_split_by_segtxt

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
batch_size = 20

def test(model, test_dataloader, type='segment', prob_mat=None, file_splits=None):

    predict_labels = []
    groundTruth_labels = []

    outputs = []

    model.eval()
    for x in test_dataloader:
        x = x.to(device)

        with torch.no_grad():
            output = model(x).to(device)
            output = output.view(-1, 48)

            outputs.extend(output.tolist())
            predict_label = torch.max(output, 1)[1]

        predict_label = predict_label.long().cpu().data.squeeze().numpy()

        predict_labels.extend(predict_label.tolist())

    if prob_mat is not None:
        # add prob mat
        predict_labels = []
        s_index = 0

        print('use prob mat')

        outputs = torch.tensor(outputs).to(device)
        for f_split in file_splits:
            f_outputs = outputs[s_index:s_index+f_split]
            f_old_predicts = predict_labels[s_index:s_index+f_split]

            for i, seg_output in enumerate(f_outputs):
                seg_prob = torch.zeros(48).to(device)
                for j, other_seg_output in enumerate(f_outputs):
                    if i != j:
                        other_seg_label = torch.max(other_seg_output, 0)[1].item()
                        seg_prob += prob_mat[other_seg_label]

                new_output = seg_output + (seg_prob / len(f_outputs))
                new_label = torch.max(new_output, 0)[1].item()
                predict_labels.append(new_label)

                if new_label != f_old_predicts[i]:
                    print(f'change {f_old_predicts[i]} to {new_label}')

            s_index += f_split


    # if frame, evaluate frame and seg acc at the same time
    if type == 'frame':
        # todo
        pass
    # if segment type, evalute seg acc directly
    elif type == 'segment':
        res = predict_labels

    return res


def get_most_frequent(arr):
    arr = np.array(arr)
    return np.argmax(np.bincount(arr.astype(np.int64)))

def get_dataloader(type='segment'):
    COMP_PATH = './'
    split = 'test'

    test_split   =  path.join(COMP_PATH, 'splits/test.split1.bundle')
    mapping_loc =  path.join(COMP_PATH, 'splits/mapping_bf.txt')
    GT_folder   =  path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
    DATA_folder =  path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos

    actions_dict = read_mapping_dict(mapping_loc)

    
    test_feat = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split, target=type) #

    if type == 'segment':
        test_dataset = TestDataset(test_feat, test_seg_txt, seed = 2)
    else:
        # todo
        pass
    
    test_dataloader = tud.DataLoader(test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers = 8, 
        pin_memory=True
    )


    return test_dataloader


test_seg_txt = './utils/test_segment.txt'

if __name__ == "__main__":

    trained_model_path = sys.argv[1]
    model_type = sys.argv[2] 

    model = torch.load(trained_model_path, map_location=torch.device(device))
    dataloader = get_dataloader(type=model_type)
    test_splits = get_file_split_by_segtxt(test_seg_txt)


    # prob_mat = torch.load('./trained/conv/prob_mat.pt', map_location=torch.device(device))

    # res = test(model, dataloader, type=model_type, prob_mat = prob_mat, file_splits = test_splits)
    res = test(model, dataloader, type=model_type)


    # save result
    path = './results/conv/test_result-0406-balance.csv'
    f = open(path, 'w+')
    f.write('Id,Category\n')

    for i, line in enumerate(res):
        f.write(f'{i},{line}\n')


    f.close()

    
