from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch
from Dataset.VideoDataset import VideoDataset, VideoTrainDataset
import torch.utils.data as tud
from Models.LSTM import LSTM_Model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import datetime
import torch.nn.utils as u
import utils.balance_data as b


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

epochs = 400
batch_size = 50

model = LSTM_Model()
learning_rate = 1e-3
log_interval = 30


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    for batch_idx, (in_feature, label, seq_length) in enumerate(train_loader):
        in_feature = in_feature.to(device)
        label = label.to(device)
        seq_length = seq_length.to(device)
        packed_input = u.rnn.pack_padded_sequence(in_feature, seq_length, batch_first=True, enforce_sorted=False).to(
            device)

        optimizer.zero_grad()
        output = model(packed_input)
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
val_dataloader = tud.DataLoader(val_dataset)

def validation(model, device, test_loader):
    model.eval()

    all_labels = []
    all_labels_predict = []
    with torch.no_grad():
        for in_feature, labels in test_loader:
            in_feature = in_feature.to(device)
            labels = labels.to(device)
            packed_input = u.rnn.pack_padded_sequence(in_feature, torch.LongTensor([in_feature.shape[1]]),
                                                      batch_first=True, enforce_sorted=False).to(device)

            output = model(packed_input)

            labels_predict = torch.max(output, 1)[1]
            all_labels.extend(labels)
            all_labels_predict.extend(labels_predict.long())

    # compute accuracy
    all_labels = torch.stack(all_labels, dim=0)
    all_labels_predict = torch.stack(all_labels_predict, dim=0)
    test_score = accuracy_score(all_labels.cpu().data.squeeze().numpy(),
                                all_labels_predict.cpu().data.squeeze().numpy())
    print('\nTest set ({:d} samples): Accuracy: {:.2f}%\n'.format(len(all_labels), 100 * test_score))

    return test_score

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

lstm = LSTM_Model(hidden_rnn_layers = 2, hidden_rnn_nodes = 368, bidirectional=True, fc_dim=256).double().to(device)
optimizer = torch.optim.Adam(list(lstm.parameters()), lr=learning_rate, weight_decay=1e-6)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_scores = []

label_dict = b.generate_label_dictionary(data_feat, data_labels)
total_segments = len(data_labels)
for epoch in range(epochs):
    new_data_feat, new_data_labels = b.balance_data(label_dict, total_segments)
    dataset = VideoTrainDataset(new_data_feat, new_data_labels, 50)
    dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_losses, train_scores = train(log_interval, lstm, device, dataloader, optimizer, epoch)

    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    if (epoch + 1) % 10 == 0:
        test_score = validation(lstm, device, val_dataloader)
        epoch_test_scores.append(test_score)

date = datetime.datetime.now().date()
torch.save(model.state_dict(), "./trained/lstm_" + date + ".pt")
A = np.array(epoch_train_losses)
B = np.array(epoch_train_scores)
D = np.array(epoch_test_scores)
np.save('./results/lstm/training_losses2_'+str(date)+'.npy', A)
np.save('./results/lstm/training_scores2_'+str(date)+'.npy', B)
np.save('./results/lstm/test_score2_'+str(date)+'.npy', D)