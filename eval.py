# eval 
from read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch
from Dataset.VideoDataset import VideoDataset
import torch.utils.data as tud
from Models.LSTM import LSTM_Model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import sys

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

model = nn.Sequential(
    nn.Linear(400, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 48)
).to(device).double()

def evaluation(trained_path):
    split = 'training'
    COMP_PATH = './'

    val_split   =  os.path.join(COMP_PATH, 'splits/val.split1.bundle')
    GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
    DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
    mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

    actions_dict = read_mapping_dict(mapping_loc)

    model.load_state_dict(torch.load(trained_path, map_location=torch.device(device)))

    # evaluate
    valid_feat, valid_labels = load_data( val_split, actions_dict, GT_folder, DATA_folder, datatype = split, target='frame') #

    valid_feat = torch.cat(valid_feat).reshape(-1, 400)
    print(len(valid_labels))

    valid_labels = torch.cat(
        [torch.Tensor(labels) for labels in valid_labels]
    ).long().flatten()

    print(valid_feat.shape, valid_labels.shape)


    val_dataset = VideoDataset(valid_feat, valid_labels)
    val_dataloader = tud.DataLoader(val_dataset, batch_size=10, shuffle=False)

    predict_labels = []
    groundTruth_labels = []

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
        
        predict_labels.extend(predict_label.tolist())
        groundTruth_labels.extend(labels.tolist())

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

    for i in range(total):
        seg_idx = segs[i]
        next_seg_idx = len(predict_labels) if i >= total-1 else segs[i+1]

        # print(seg_idx, next_seg_idx)
        seg_predict_label = get_most_frequent(predict_labels[seg_idx:next_seg_idx])

        if seg_predict_label == results[i]:
            correct_nums += 1

    print('valid seg acc:', correct_nums / total)

def get_most_frequent(arr):
    arr = np.array(arr)
    return np.argmax(np.bincount(arr.astype(np.int64)))

if __name__ == "__main__":

    trained_model_path = sys.argv[1]

    print(trained_model_path)
    evaluation(trained_model_path)
