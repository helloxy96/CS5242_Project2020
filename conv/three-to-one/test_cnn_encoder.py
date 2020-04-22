# eval 
from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os.path as path
import numpy as np
import torch

from Dataset.EncoderTestDataset import TestDataset

import torch.utils.data as tud
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
from utils.data import get_file_split_by_segtxt, get_max_prob_seg_seq

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
batch_size = 1

def test(models, test_dataloader, type='segment', prob_mat=None, h0_vec=None, file_splits=None):

    predict_1_model, encoder, single_classifier, three_classifier = models

    predict_labels = []
    groundTruth_labels = []

    outputs = []

    predict_1_model.eval()
    encoder.eval()
    single_classifier.eval()
    three_classifier.eval()
    for x in test_dataloader:
        x = x.to(device)

        with torch.no_grad():
            if x.shape[1] <= 1:
                x = x.squeeze(dim=0)
                output = predict_1_model(x).to(device)
            elif x.shape[1] == 3:
                before_logits = encoder(x[:, 0])
                logits = encoder(x[:, 1])
                next_logits = encoder(x[:, 2])

                predict_1 = single_classifier(logits)

                all_logits = torch.cat([before_logits, logits, next_logits], dim=1)
                predict_2 = three_classifier(all_logits)

                output = predict_1 + predict_2

            output = output.view(-1, 48)

            outputs.extend(output.tolist())
            predict_label = torch.max(output, 1)[1]

        predict_label = predict_label.long().cpu().data.squeeze().item()

        predict_labels.append(predict_label)
        # print(predict_label)

    new_predict_labels = []
    if prob_mat is not None:
        # add prob mat
        
        s_index = 0

        print('use prob mat')

        outputs = torch.tensor(outputs).to(device)
        for f_split in file_splits:
            f_outputs = outputs[s_index:s_index+f_split]
            f_old_predicts = predict_labels[s_index:s_index+f_split]

            new_label = get_max_prob_seg_seq(f_outputs, prob_mat, h0_vec) 


            if new_label != f_old_predicts:
                print(f'change {f_old_predicts} to {new_label}')

            s_index += f_split

            new_predict_labels.extend(new_label)

    # if frame, evaluate frame and seg acc at the same time
    if type == 'frame':
        # todo
        pass
    # if segment type, evalute seg acc directly
    elif type == 'segment':
        if len(new_predict_labels) > 0:
            res = new_predict_labels
        else:
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
        test_dataset = TestDataset(test_feat, test_seg_txt, seed = 2, chop_num=3)
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

    predict_1_model = torch.load(trained_model_path, map_location=torch.device(device))
    
    encoder = torch.load('./trained/cnn_encoder/encoder_2020-04-11-12_.pkl', map_location=torch.device(device))
    single_classifier = torch.load('./trained/cnn_encoder/single_classifier_2020-04-11-12_.pkl', map_location=torch.device(device))
    three_classifier = torch.load('./trained/cnn_encoder/three_classifier_2020-04-11-12_.pkl', map_location=torch.device(device))

    
    dataloader = get_dataloader(type=model_type)
    test_splits = get_file_split_by_segtxt(test_seg_txt)


    prob_mat = torch.load('./trained/conv/prob_mat_next.pt', map_location=torch.device(device))
    h0_vec = torch.load('./trained/conv/prob_h0.pt', map_location=torch.device(device))

    res = test(
        [predict_1_model, encoder, single_classifier, three_classifier], 
        dataloader, type=model_type, prob_mat = prob_mat, h0_vec = h0_vec, file_splits = test_splits
    )
    # res = test(model, dataloader, type=model_type)


    # save result
    path = './results/cnn_encoder/test_result-0411-encoder-prob.csv'
    f = open(path, 'w+')
    f.write('Id,Category\n')

    for i, line in enumerate(res):
        f.write(f'{i},{line}\n')


    f.close()

    
