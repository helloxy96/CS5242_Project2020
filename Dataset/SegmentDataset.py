import torch.utils.data as tud
import torch
import math
import random
import copy

class SegmentDataset(tud.Dataset):
    def __init__(self, raw_segs, raw_labels, seed=None, chop_num=0):
        super(SegmentDataset, self).__init__()
        self.raw_segs = raw_segs
        self.raw_labels = raw_labels

        self.random_seed = seed

        self.labels = []
        self.segments = []

        self.segments, self.labels = self.process(raw_segs, raw_labels)

    def __len__(self):
        return len(self.segments)

    def process(self, segs, data_labels):

        x = []
        labels = []

        for i, feat in enumerate(segs):
            # remove silent segment
            if data_labels[i] == 0:
                continue 

            # less than 5 frames, skip
            if len(feat) < 5:
                print('ignore', i, len(feat))
                continue

            # take pieces from segment
            # get 100 pieces
            # a piece with fixed length, 1
            piece_len = 1
            piece_segs = 100

            # < 100 need repeat to extend
            while len(feat) < piece_len * piece_segs:
                feat = feat.repeat(2, 1)
            # feat = feat.repeat(math.ceil(len(feat) / piece_segs) + 1, 1)

            step = math.floor(len(feat) / piece_segs)

            # random sample from a seg
            seg_range = range(len(feat)-piece_len + 1)

            if self.random_seed:
                random.seed(self.random_seed)
            indexes = random.sample(seg_range, piece_segs)
            indexes.sort()

            item = []
            for s_i in indexes:
                item.append(feat[s_i:s_i+piece_len])
            
            x.append(item)

            labels.append(data_labels[i])
        
        # print(labels)
        return x, labels

    def __getitem__(self, idx):
        # print(len(self.segments[idx]), len(self.segments[idx][0]))
        segment = torch.cat(self.segments[idx]).double()
        label = torch.tensor(self.labels[idx]).long()

        # transform
        segment = segment.permute(1, 0).contiguous()

        return segment, label