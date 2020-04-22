#!/usr/bin/env python
# coding: utf-8

# In[24]:


import torch
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
# cerealsModel = torch.load("./trained/tasks/cereals_epoch20_2020-04-07.pkl", map_location=torch.device(device))
# coffeeModel = torch.load("./trained/tasks/coffee_epoch30_2020-04-07.pkl", map_location=torch.device(device))
# friedeggModel = torch.load("./trained/tasks/friedegg_epoch30_2020-04-07.pkl", map_location=torch.device(device))
# juiceModel = torch.load("./trained/tasks/juice_epoch40_2020-04-07.pkl", map_location=torch.device(device))
# milkModel = torch.load("./trained/tasks/milk_epoch40_2020-04-07.pkl", map_location=torch.device(device))
# pancakeModel = torch.load("./trained/tasks/pancake_epoch40_2020-04-07.pkl", map_location=torch.device(device))
# salatModel = torch.load("./trained/tasks/salat_epoch30_2020-04-07.pkl", map_location=torch.device(device))
# sandwichModel = torch.load("./trained/tasks/sandwich_epoch10_2020-04-07.pkl", map_location=torch.device(device))
# scrambledeggModel = torch.load("./trained/tasks/scrambledegg_epoch40_2020-04-07.pkl", map_location=torch.device(device))
# teaModel = torch.load("./trained/tasks/tea_epoch10_2020-04-07.pkl", map_location=torch.device(device))


import glob

conv_file_paths = glob.glob("./trained/conv_1_to_1/task_0414/*.pkl")
cnn_file_paths = glob.glob("./trained/conv_3_to_1/tasks_0411/*.pkl")

# modelDictionary = {
#     "cereals" : [cerealsModel],
#     "coffee" : [coffeeModel],
#     "friedegg" : [friedeggModel],
#     "juice" : [juiceModel],
#     "milk" : [milkModel],
#     "pancake" : [pancakeModel],
#     "salat" : [salatModel],
#     "sandwich" : [sandwichModel],
#     "scrambledegg" : [scrambledeggModel],
#     "tea" : [teaModel]
# }

modelDictionary = {
    "cereals" : [],
    "coffee" : [],
    "friedegg" : [],
    "juice" : [],
    "milk" : [],
    "pancake" : [],
    "salat" : [],
    "sandwich" : [],
    "scrambledegg" : [],
    "tea" : []
}

# add other models in it
import re
for key, model_list in modelDictionary.items():
    for model_name in ['conv', 'encoder', 'single_classifier', 'three_classifier']:
        r = re.compile(f'.*{model_name}_{key}.*')

        if model_name == 'conv':
            file_name = list(filter(r.match, conv_file_paths))[0]        
        else:
            file_name = list(filter(r.match, cnn_file_paths))[0]

        m = torch.load(file_name, map_location=torch.device(device))
        model_list.append(m)

print(len(modelDictionary['milk']))


# In[1]:


from utils.read_datasetBreakfast import load_testdata
import os

COMP_PATH = ''

test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
data_feat = load_testdata(test_split, DATA_folder)


# In[8]:
test_seg_txt = './utils/test_segment.txt'


segment_idx = []
with open("./utils/test_segment.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        segment_idx.append(line[:-1].split(" "))


# In[18]:


data_feat_seg = []
for i in range(len(data_feat)):
    videoFrames = []
    for j in range(len(segment_idx[i])-1):
        videoFrames.append(data_feat[i][1][int(segment_idx[i][j]):int(segment_idx[i][j+1])])
    data_feat_seg.append((data_feat[i][0], videoFrames))


# In[42]:


import numpy as np
from Dataset.EncoderTestDataset import TestDataset
import torch.utils.data as tud
from utils.data import get_file_split_by_segtxt, get_max_prob_seg_seq


prob_mat = torch.load('./trained/prob_mat_next.pt', map_location=torch.device(device))
h0_vec = torch.load('./trained/prob_h0.pt', map_location=torch.device(device))

test_splits = get_file_split_by_segtxt(test_seg_txt)

batch_size = 1
outputs = []
predict_labels = []
for i in data_feat_seg:
    task = i[0]

    test_dataset = TestDataset([i[1]], seed = 2, chop_num=3)
    test_dataloader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    models = modelDictionary[task]
    models = [m.to(device) for m in models]
    predict_1_model, encoder, single_classifier, three_classifier = models
    
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

new_predict_labels = []
if prob_mat is not None:
    # add prob mat
    
    s_index = 0

    print('use prob mat')

    outputs = torch.tensor(outputs).to(device)
    for f_split in test_splits:
        f_outputs = outputs[s_index:s_index+f_split]
        f_old_predicts = predict_labels[s_index:s_index+f_split]

        new_label = get_max_prob_seg_seq(f_outputs, prob_mat, h0_vec) 


        if new_label != f_old_predicts:
            print(f'change {f_old_predicts} to {new_label}')

        s_index += f_split

        new_predict_labels.extend(new_label)


if len(new_predict_labels) > 0:
    res = new_predict_labels
else:
    res = predict_labels


# In[41]:


print(len(res))
            


# In[43]:


path = './results/final/test_result_bytask_0414.csv'
f = open(path, 'w+')
f.write('Id,Category\n')

counter = 0
for i in res:
        f.write(f'{counter},{i}\n')
        counter += 1
f.close()

