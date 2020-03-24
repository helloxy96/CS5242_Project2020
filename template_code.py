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
train_split =  os.path.join(COMP_PATH, 'splits/dev_train.split1.bundle') #Train Split
test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video 
DATA_folder =  os.path.join(COMP_PATH, 'Data/') #Frame I3D features for all videos
mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt') 

actions_dict = read_mapping_dict(mapping_loc)
if  split == 'training':
    data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
if  split == 'test':
    data_feat = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features only

'''
Write Code Below
Pointers
Need to load the segments.txt file for segments for test videos 
Output the CSV in correct format as shown in Evaluation Section
Id corresponds to the segments in order. 
Example - 30-150 = Id 0
          150-428 = Id 1
          428-575 = Id 2
Category is the Class of the Predicted Action
'''
print(len(data_feat))
print(data_feat[0].shape)
print(len(data_labels))
print(data_labels[0])

