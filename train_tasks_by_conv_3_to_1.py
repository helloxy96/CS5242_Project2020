
from utils.read_datasetBreakfast import load_data, read_mapping_dict
import os
import numpy as np
import torch
from Dataset.FileDataset import FileDataset

import torch.utils.data as tud
from torch.utils.data.sampler import WeightedRandomSampler

from Models.Conv_3_to_1 import encoder, single_logits_classifier, three_logits_classifier
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
from evaluation import evaluation
from datetime import datetime
import utils.balance_data as b
import random
import sys

TASK_NAME = sys.argv[1]

torch.cuda.set_device(1)

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

data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split, target='file') #
# valid_feat, valid_labels = load_data( val_split, actions_dict, GT_folder, DATA_folder, datatype = split, target='file') #

# balance data
# label_dict = b.generate_label_dictionary(data_feat, data_labels)
# total_segments = len(data_labels)
# data_feat, data_labels = b.balance_data(label_dict, total_segments)

# select data
print(len(data_feat))
train_data_feat = data_feat[:-15]
train_data_label = data_labels[:-15]
print(len(train_data_feat))

val_data_feat = data_feat[-15:]
val_data_label = data_labels[-15:]

# # model part
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

# for loss weight of unbalanced data
# loss_weights = torch.ones(48)
# for label in data_labels:
#     loss_weights[label] += 1

# normalization = torch.sum(loss_weights).item()
# loss_weights = ((normalization) / loss_weights).double()

# use weighted sampler
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
epochs = 30
batch_size = 20

learning_rate = 1e-3
log_interval = 10
valid_interval = 3 # 5 epoch a check

print(len(data_feat), len(data_labels))
print(len(data_feat[0]), data_labels[0])
print(data_feat[0][0].shape)
# dataset
dataset = FileDataset(train_data_feat, train_data_label, chop_num = 3)
train_loader = tud.DataLoader(dataset, 
    batch_size=batch_size, 
    shuffle = True,
    num_workers = 8, 
    pin_memory=True
)

print(len(dataset))
print(dataset[0][0].shape, dataset[0][1].shape)

valid_dataset = FileDataset(val_data_feat, val_data_label, seed = 2, chop_num = 3)
valid_loader = tud.DataLoader(valid_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers = 8, 
    pin_memory=True
)


# model and optimizer
# encoder = Encoder(n_inputs=150)
encoder, single_logits_classifier, three_logits_classifier = [
    model.to(device).double()
    for model in [
        encoder, 
        single_logits_classifier, 
        three_logits_classifier
    ]
]

parameters = [
    *list(encoder.parameters()),
    *list(single_logits_classifier.parameters()),
    *list(three_logits_classifier.parameters()),
]
optimizer = torch.optim.Adam(parameters, lr=learning_rate)


def train(epoch):
    encoder.train()
    single_logits_classifier.train()
    three_logits_classifier.train()

    losses = []
    scores = []
    for batch_idx, (in_feature, label) in enumerate(train_loader):
        in_feature = in_feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # in_feature (batch_size, seg_num, 400, 100)
        if in_feature.shape[1] < 3:
            in_feature = in_feature.squeeze(dim=1)
            logits = encoder(in_feature[0])
            predict_1 = single_logits_classifier(logits)
            loss = F.cross_entropy(predict_1, label)
            output = predict_1
        elif in_feature.shape[1] == 3:

            before_logits = encoder(in_feature[:, 0])
            logits = encoder(in_feature[:, 1])
            next_logits = encoder(in_feature[:, 2])

            predict_1 = single_logits_classifier(logits)

            all_logits = torch.cat([before_logits, logits, next_logits], dim=1)
            predict_2 = three_logits_classifier(all_logits)

            loss_1 = F.cross_entropy(predict_1, label)
            loss_2 = F.cross_entropy(predict_2, label)

            loss = loss_1 + loss_2
            output = predict_1 + predict_2

        loss.backward()
        optimizer.step()

        #step_score = accuracy_score(label.cpu().data.squeeze().numpy(), label_predict.cpu().data.squeeze().numpy())
        # bottleneck? 
        with torch.no_grad():
            label_predict = torch.max(output, 1)[1]
            correct = (label_predict == label).sum().item()
            step_score = correct / len(label)
            scores.append(step_score)
            losses.append(loss.item())


        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, (batch_idx + 1) * batch_size, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores

def evaluate():
    encoder.eval()
    single_logits_classifier.eval()
    three_logits_classifier.eval()

    valid_corrects = []

    valid_losses = []
    valid_scores = []
    for batch_idx, (in_feature, label) in enumerate(valid_loader):
        in_feature = in_feature.to(device)
        label = label.to(device).flatten()

        with torch.no_grad():
        # in_feature (batch_size, seg_num, 400, 100)
            if in_feature.shape[1] < 3:
                in_feature = in_feature.squeeze(dim=1)
                logits = encoder(in_feature)
                predict_1 = single_logits_classifier(logits)

                loss = F.cross_entropy(predict_1, label)
                output = predict_1
            elif in_feature.shape[1] == 3:

                before_logits = encoder(in_feature[:, 0])
                logits = encoder(in_feature[:, 1])
                next_logits = encoder(in_feature[:, 2])

                predict_1 = single_logits_classifier(logits)

                all_logits = torch.cat([before_logits, logits, next_logits], dim=1)
                predict_2 = three_logits_classifier(all_logits)

                loss_1 = F.cross_entropy(predict_1, label)
                loss_2 = F.cross_entropy(predict_2, label)

                loss = loss_1 + loss_2
                output = predict_1 + predict_2

            label_predict = torch.max(output, 1)[1]
            valid_corrects.append((label_predict == label).sum().item())
            valid_losses.append(loss.item())


    return  sum(valid_losses) / len(valid_loader) , sum(valid_corrects) / len(valid_dataset)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_scores = []
epoch_test_losses = []

dt = datetime.now()
timestamp = str(dt.date()) + '-' +str(dt.hour)

maxValidationAcc = 0

for epoch in range(epochs):
    # print('begin balance')        
    # new dataset

    # 30 pick 1
    seed = random.randint(0, 30)
    dataset = FileDataset(train_data_feat, train_data_label, seed=seed, chop_num = 3)
    train_loader = tud.DataLoader(dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers = 8, 
        pin_memory=True
    )

    train_losses, train_scores = train(epoch)

    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)

    # validation
    if (epoch + 1) % valid_interval == 0:
        valid_loss, valid_scores = evaluate()

        epoch_test_losses.append(valid_loss)
        epoch_test_scores.append(valid_scores)

        if valid_scores > maxValidationAcc:
            maxValidationAcc = valid_scores
            torch.save(encoder, f'./trained/conv_3_to_1/tasks/encoder_{TASK_NAME}_{timestamp}.pkl')
            torch.save(single_logits_classifier, f'./trained/conv_3_to_1/tasks/single_classifier_{TASK_NAME}_{timestamp}.pkl')
            torch.save(three_logits_classifier, f'./trained/conv_3_to_1/tasks/three_classifier_{TASK_NAME}_{timestamp}.pkl')

        print('valid loss:', valid_loss)
        print('valid scores:', valid_scores)



A = np.array(epoch_train_losses)
B = np.array(epoch_train_scores)

C = np.array(epoch_test_losses)
D = np.array(epoch_test_scores)

np.save(f'./results/conv_3_to_1/tasks/training_losses_{TASK_NAME}_{timestamp}.npy', A)
np.save(f'./results/conv_3_to_1/tasks/training_scores_{TASK_NAME}_{timestamp}.npy', B)

np.save(f'./results/conv_3_to_1/tasks/test_losses_{TASK_NAME}_{timestamp}.npy', C)
np.save(f'./results/conv_3_to_1/tasks/test_scores_{TASK_NAME}_{timestamp}.npy', D)




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

