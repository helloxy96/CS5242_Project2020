from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch
from Dataset.VideoDataset import VideoDataset
import torch.utils.data as tud
from Models.LSTM import LSTM_Model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import datetime


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
validation_split = os.path.join(COMP_PATH, 'splits/val.split1.bundle') #Validation split
val_data_feat, val_data_labels = load_data( validation_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels


def splitByFrames(data_feat, data_labels, frames_per_clip):
    new_data_feat = []
    new_data_labels = []
    for i in range(0, len(data_feat)):
        total_frames, dim = data_feat[i].shape
        segment = data_feat[i]
        if total_frames % frames_per_clip != 0:
            pad_num = frames_per_clip - total_frames % frames_per_clip
            pad_Tensor = torch.zeros((pad_num, 400), dtype=torch.float64)
            segment = torch.cat((segment, pad_Tensor))
        total_frames, dim = segment.shape
        clip_num = int(total_frames / frames_per_clip)
        clips = segment.view(-1, frames_per_clip, 400)

        label = data_labels[i]
        new_data_labels += [label] * clip_num
        new_data_feat.append(clips)

    new_data_feat = torch.cat(new_data_feat)
    return new_data_feat, new_data_labels


new_data_feat, new_data_labels = splitByFrames(data_feat, data_labels, 16)
val_data_feat, val_data_labels = splitByFrames(val_data_feat, val_data_labels, 16)

epochs = 100
batch_size = 50

dataset = VideoDataset(new_data_feat, new_data_labels)
dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = LSTM_Model()
learning_rate = 1e-3
log_interval = 200


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
        step_score = accuracy_score(label.cpu().data.squeeze().numpy(), label_predict.cpu().data.squeeze().numpy())
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, (batch_idx + 1) * batch_size, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


val_dataset = VideoDataset(val_data_feat, val_data_labels)
val_dataloader = tud.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def validation(model, device, test_loader):
    model.eval()

    test_loss = 0
    all_labels = []
    all_labels_predict = []
    with torch.no_grad():
        for in_feature, labels in test_loader:
            in_feature = in_feature.to(device)
            labels = labels.to(device)

            output = model(in_feature)
            loss = F.cross_entropy(output, labels)
            test_loss += loss.item()
            labels_predict = torch.max(output, 1)[1]
            all_labels.extend(labels)
            all_labels_predict.extend(labels_predict)

    test_loss = test_loss / len(test_loader.dataset)
    # compute accuracy
    all_labels = torch.stack(all_labels, dim=0)
    all_labels_predict = torch.stack(all_labels_predict, dim=0)
    test_score = accuracy_score(all_labels.cpu().data.squeeze().numpy(),
                                all_labels_predict.cpu().data.squeeze().numpy())
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_labels), test_loss,
                                                                                        100 * test_score))

    return test_loss, test_score


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

lstm = LSTM_Model().double().to(device)
optimizer = torch.optim.Adam(list(lstm.parameters()), lr=learning_rate)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

for epoch in range(epochs):
    train_losses, train_scores = train(log_interval, lstm, device, dataloader, optimizer, epoch)
    test_losses, test_score = validation(lstm, device, val_dataloader)

    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(test_losses)
    epoch_test_scores.append(test_score)

torch.save(model.state_dict(), "./trained/lstm.pt")
A = np.array(epoch_train_losses)
B = np.array(epoch_train_scores)
C = np.array(epoch_test_losses)
D = np.array(epoch_test_scores)
date = datetime.datetime.now().date()
np.save('./results/lstm/training_losses_'+str(date)+'.npy', A)
np.save('./results/lstm/training_scores_'+str(date)+'.npy', B)
np.save('./results/lstm/test_loss_'+str(date)+'.npy', C)
np.save('./results/lstm/test_score_'+str(date)+'.npy', D)