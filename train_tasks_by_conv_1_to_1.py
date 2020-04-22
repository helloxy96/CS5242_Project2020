
from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch
from Dataset.SegmentDataset import SegmentDataset

import torch.utils.data as tud
from torch.utils.data.sampler import WeightedRandomSampler

from Models.Conv import Encoder
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
from conv.evaluation import evaluation
from datetime import datetime
import utils.balance_data as b
import random
import sys

TASK_NAME = sys.argv[1]

torch.cuda.set_device(2)

COMP_PATH = ''

''' 
training to load train set
test to load test set
'''
split = 'training'
#split = 'test'
train_split =  os.path.join(COMP_PATH, f'splits/{TASK_NAME}.split.bundle') #Train Split
# train_split =  os.path.join(COMP_PATH, 'splits/dev_train.split1.bundle') #Train Split
val_split   =  os.path.join(COMP_PATH, 'splits/val.split1.bundle')
test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

actions_dict = read_mapping_dict(mapping_loc)

data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #
# valid_feat, valid_labels = load_data( val_split, actions_dict, GT_folder, DATA_folder, datatype = split) #

print(len(data_feat))
train_data_feat = data_feat[:-51]
train_data_label = data_labels[:-51]
print(len(train_data_feat))

valid_feat = data_feat[-51:]
valid_labels = data_labels[-51:]

# try balance data
# label_dict = b.generate_label_dictionary(data_feat, data_labels)
# total_segments = len(data_labels)
# data_feat, data_labels = b.balance_data(label_dict, total_segments)


# # model part
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

# try loss weight of unbalanced data
# loss_weights = torch.ones(48)
# for label in data_labels:
#     loss_weights[label] += 1

# normalization = torch.sum(loss_weights).item()
# loss_weights = ((normalization) / loss_weights).double()


# try use weighted sampler
# weights = torch.ones(48)
# for label in data_labels:
#     weights[label] += 1

# weights = (1. / weights).double().to(device)

# sampler = WeightedRandomSampler(
#     weights=weights,
#     num_samples=len(data_feat),
#     replacement=True
# )


# train parameters
epochs = 40
batch_size = 20

learning_rate = 1e-3
log_interval = 50
valid_interval = 3 # 5 epoch a check

print(len(train_data_feat), len(train_data_label))
# dataset
dataset = SegmentDataset(train_data_feat, train_data_label)
dataloader = tud.DataLoader(dataset, 
    batch_size=batch_size, 
    shuffle = True,
    num_workers = 8, 
    pin_memory=True
)

valid_dataset = SegmentDataset(valid_feat, valid_labels, seed = 2)
valid_dataloader = tud.DataLoader(valid_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers = 8, 
    pin_memory=True
)



# model and optimizer
model = Encoder(n_inputs = 100)
model = model.to(device).double()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    for batch_idx, (in_feature, label) in enumerate(train_loader):
        in_feature = in_feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(in_feature)
        output = output.view(-1, 48)
        label = label.flatten()
        loss = F.cross_entropy(output, label)
        losses.append(loss.item())
        
        #step_score = accuracy_score(label.cpu().data.squeeze().numpy(), label_predict.cpu().data.squeeze().numpy())
        # bottleneck? 
        with torch.no_grad():
            label_predict = torch.max(output, 1)[1]
            correct = (label_predict == label).sum().item()
            step_score = correct / len(label)
            scores.append(step_score)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, (batch_idx + 1) * batch_size, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_frame_scores = []
epoch_test_seg_scores = []

maxValidationAcc = 0
dt = datetime.now()
timestamp = str(dt.date()) + '-' +str(dt.hour)

for epoch in range(epochs):
    # print('begin balance')        
    # new dataset

    # 60 pick 1
    seed = random.randint(0, 30)
    dataset = SegmentDataset(train_data_feat, train_data_label, seed=seed)
    dataloader = tud.DataLoader(dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers = 8, 
        pin_memory=True
    )

    train_losses, train_scores = train(log_interval, model, device, dataloader, optimizer, epoch)

    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)

    # validation
    if (epoch + 1) % valid_interval == 0:
        res = evaluation(model, valid_dataloader, type='segment')

        if res['seg']['acc'] > maxValidationAcc:
            maxValidationAcc = res['seg']['acc']
            torch.save(model, f'./trained/conv_1_to_1/task/conv_{TASK_NAME}_{timestamp}.pkl')

        # frame result may not exist
        if 'frame' in res:
            epoch_test_frame_scores.append(res['frame']['acc'])
        epoch_test_seg_scores.append(res['seg']['acc'])

A = np.array(epoch_train_losses)
B = np.array(epoch_train_scores)

C = np.array(epoch_test_frame_scores)
D = np.array(epoch_test_seg_scores)

np.save(f'./results/conv_1_to_1/task/training_{TASK_NAME}_losses_{timestamp}.npy', A)
np.save(f'./results/conv_1_to_1/task/training_{TASK_NAME}_scores_{timestamp}.npy', B)

np.save(f'./results/conv_1_to_1/task/test_{TASK_NAME}_frame_scores_{timestamp}.npy', C)
np.save(f'./results/conv_1_to_1/task/test_{TASK_NAME}_seg_scores_{timestamp}.npy', D)




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

