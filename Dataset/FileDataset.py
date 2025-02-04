import torch.utils.data as tud
import torch
import math
import random
import copy

class FileDataset(tud.Dataset):
    def __init__(self, raw_segs, raw_labels, seed=None, chop_num=0, set_type='train'):
        super(FileDataset, self).__init__()
        self.raw_segs = raw_segs
        self.raw_labels = raw_labels

        self.random_seed = seed
        self.chop_num = chop_num
        self.set_type = set_type

        self.labels = []
        self.x = []
        self.x, self.labels = self.process(raw_segs, raw_labels)

        if self.chop_num > 0:
            # chop into segs
            self.x, self.labels = self.chop_to_segs(self.x, self.labels)
            

    def __len__(self):
        return len(self.x)

    def chop_to_segs(self, files, file_labels):
        x = []
        labels = []
        chop_num = self.chop_num

        for i, file in enumerate(files):
            if len(file) < chop_num:
                continue

            f_labels = file_labels[i]
            
            for j in range(0, len(file)-chop_num+1):         
                x.append(file[j:j+chop_num])
                labels.append(f_labels[j+chop_num//2])

        return x, labels

    def process(self, files, file_labels):

        # files.shape: file_nums, seg nums in each file, 400
        # file_labels.shape: file_nums, seg nums in each file
        x = []
        labels = []

        for i, file in enumerate(files):
            f_x = []
            f_labels = []

            for j, seg in enumerate(file):

                feat_label = file_labels[i][j]
                feats = seg

                # remove silent segment
                if feat_label == 0:
                    continue 

                # less than 1s, skip
                if len(feats) < 5:
                    print('ignore', i, len(feats))
                    continue

                # take pieces from segment
                # get 50 pieces
                # a piece with fixed length, 1
                piece_len = 1
                piece_segs = 100

                # < 50 need repeat to extend
                while len(feats) < piece_len * piece_segs:
                    feats = feats.repeat(2, 1)
                # feat = feat.repeat(math.ceil(len(feat) / piece_segs) + 1, 1)

                step = math.floor(len(feats) / piece_segs)

                # random sample from a seg
                seg_range = range(len(feats)-piece_len + 1)

                if self.random_seed:
                    random.seed(self.random_seed)
                indexes = random.sample(seg_range, piece_segs)
                indexes.sort()

                item = []
                for s_i in indexes:
                    item.append(feats[s_i:s_i+piece_len].tolist())
                
                # item = []

                # for s_i in range(piece_segs):
                #     item.append(feat[s_i*step:s_i*step+piece_len])
                
                f_x.append(item)
                f_labels.append(feat_label)

            if len(f_x) > 0:
                x.append(f_x)
                labels.append(f_labels)
        
        # print(labels)
        return x, labels

    def __getitem__(self, idx):
        # print(len(self.files[idx]), len(self.files[idx][0]))
        file = torch.tensor(self.x[idx]).double()
        labels = torch.tensor(self.labels[idx]).long()

        # transform
        # file would be segs, 400, 50
        try:
            file = file.squeeze(dim = 2)
            file = file.permute(0, 2, 1).contiguous()
        except:
            print('exception file idx', idx)

        # labels: segs

        return file, labels