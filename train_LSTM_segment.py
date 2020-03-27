#!/usr/bin/env python
# coding: utf-8

# In[3]:


from read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np


COMP_PATH = ''

''' 
training to load train set
test to load test set
'''
split = 'training'
#split = 'test'
train_split =  os.path.join(COMP_PATH, 'splits/train.exclude_val.bundle') #Train Split
test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video 
DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt') 

actions_dict = read_mapping_dict(mapping_loc)

data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels


# In[7]:


segment_idx = []
with open("training_segment.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        segment_idx.append(line[:-1].split(" "))


# In[9]:


new_data_feat = []
new_data_labels = []
for i in range(len(data_feat)):
    idx = segment_idx[i]
    for x in range(len(idx) - 1):
        new_data_feat.append(data_feat[i][int(idx[x]) : int(idx[x+1])])
        new_data_labels.append((data_labels[i][x]))

#new_data_feat: list of Tensor with shape (frames, 400), len is total segments num
#new_data_labels: list of int


# In[36]:


import torch
from Dataset.VideoDataset import VideoDataset
import torch.utils.data as tud
from Models.LSTM import LSTM_Model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

epochs = 10
#batch_size = 50

dataset = VideoDataset(new_data_feat, new_data_labels)
dataloader = tud.DataLoader(dataset, shuffle=True)

model = LSTM_Model()
learning_rate = 1e-3
log_interval = 10000

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    
    losses = []
    scores = []
    for batch_idx, (in_feature, label) in enumerate(train_loader):
        in_feature = in_feature.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(in_feature)
        loss = F.cross_entropy(output, label)
        losses.append(loss.item())
        
        label_predict = torch.max(output, 1)[1]
        step_score = accuracy_score(label.cpu().data.numpy(), label_predict.cpu().data.numpy())
        scores.append(step_score)
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, (batch_idx+1), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        
    return losses, scores


# In[37]:


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

lstm = LSTM_Model().double().to(device)
optimizer = torch.optim.Adam(list(lstm.parameters()), lr=learning_rate, weight_decay=1e-6)

# record training process
epoch_train_losses = []
epoch_train_scores = []
for epoch in range(epochs):
    train_losses, train_scores = train(log_interval, lstm, device, dataloader, optimizer, epoch)
    
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)

torch.save(model.state_dict(), "./trained/lstm.pt")
A = np.array(epoch_train_losses)
B = np.array(epoch_train_scores)
np.save('./log/training_losses_2.npy', A)
np.save('./log/training_scores_2.npy', B)


# In[ ]:


# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
# plt.title("model loss")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['train'], loc="upper left")
# # 2nd figure
# plt.subplot(122)
# plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
# plt.title("training scores")
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(['train'], loc="upper left")
# plt.show()

