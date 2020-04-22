
from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch
from Dataset.VideoDataset import VideoDataset
import torch.utils.data as tud
from Models.MLP import model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
from evaluation import evaluation
from datetime import datetime
import utils.balance_data as b

torch.cuda.set_device(2)

COMP_PATH = ''

''' 
training to load train set
test to load test set
'''
split = 'training'
#split = 'test'
train_split =  os.path.join(COMP_PATH, 'splits/train.exclude_val.bundle') #Train Split
# train_split =  os.path.join(COMP_PATH, 'splits/dev_train.split1.bundle') #Train Split
val_split   =  os.path.join(COMP_PATH, 'splits/val.split1.bundle')
test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

actions_dict = read_mapping_dict(mapping_loc)

data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split, target='frame') #


x = torch.cat(data_feat).reshape(-1, 400)
print(len(data_labels))

labels = torch.cat(
    [torch.Tensor(labels) for labels in data_labels]
).long().flatten()


print(x.shape, labels.shape)

# filter 'silent' data
mask = labels == 0
labels = labels[~mask]
x = x[~mask] 

print(x.shape, labels.shape)

# # model part
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

epochs = 30
batch_size = 100

# dataset = VideoDataset(x, labels)
# dataloader = tud.DataLoader(dataset, 
#     batch_size=batch_size, 
#     shuffle=True, 
#     num_workers = 8, 
#     pin_memory=True
# )

learning_rate = 1e-3
log_interval = 1000
valid_interval = 3 # 3 epoch a check


model = model.to(device).double()

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=1e-6)


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

label_dict = b.generate_label_dictionary(x.tolist(), labels.tolist())
total_segments = len(labels)

for epoch in range(epochs):
    # print('begin balance')
    new_data_feat, new_data_labels = b.balance_data(label_dict, total_segments)
    
    new_data_feat = torch.tensor(new_data_feat).double()
    new_data_labels = torch.tensor(new_data_labels).long()
    
    dataset = VideoDataset(new_data_feat, new_data_labels)
    dataloader = tud.DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers = 8, 
            pin_memory = True
        )
    # print('end balance and data setup')

    train_losses, train_scores = train(log_interval, model, device, dataloader, optimizer, epoch)

    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)

    # validation
    if (epoch + 1) % valid_interval == 0:
        res = evaluation(model)

        epoch_test_frame_scores.append(res['frame']['acc'])
        epoch_test_seg_scores.append(res['seg']['acc'])

A = np.array(epoch_train_losses)
B = np.array(epoch_train_scores)

C = np.array(epoch_test_frame_scores)
D = np.array(epoch_test_seg_scores)

timestamp = datetime.now().date()

np.save(f'./results/fc/training_losses_fc_{timestamp}.npy', A)
np.save(f'./results/fc/training_scores_fc_{timestamp}.npy', B)

np.save(f'./results/fc/test_frame_scores_{timestamp}.npy', C)
np.save(f'./results/fc/test_seg_scores_{timestamp}.npy', D)

torch.save(model, f'./trained/fc_devtrain_{timestamp}.pkl')



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

