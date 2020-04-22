import torch.utils.data as tud
import torch
import math
import random
import copy


class TestDataset(tud.Dataset):
    def __init__(self, raw_data, seg_info_path, seed = None):
        super(TestDataset, self).__init__()
        
        self.raw_data = raw_data
        self.seg_info_path = seg_info_path

        self.random_seed = seed

        self.x = self.get_seg_x(raw_data, seg_info_path)
        self.x = self.process(self.x)

    def __len__(self):
        return len(self.x)

    def get_seg_x(self, raw_data, seg_info_path):
        x = []        
        f = open(seg_info_path, 'r')

        for f_i, line in enumerate(f):

            if f_i >= len(raw_data):
                break

            indexes = line.split()

            for i in range(len(indexes)-1):
                begin_index, end_index = int(indexes[i]), int(indexes[i+1])
                x.append(raw_data[f_i][begin_index:end_index])
            

        return x

    def process(self, raw_data):

        x = []
        segs = raw_data

        for i, feat in enumerate(segs):

            # less than 1s, skip
            if len(feat) < 5:
                print('ignore!', len(feat))
                continue

           # take pieces from segment
            # get 50 pieces
            # a piece with fixed length, 1
            piece_len = 1
            piece_segs = 50

            # < 50 need copy to extend
            while len(feat) < piece_len * piece_segs:
                feat = feat.repeat(2, 1)


            step = math.floor(len(feat) / piece_segs)

            # random sample from a seg
            seg_range = range(len(feat)-piece_len + 1)

            # fix random seed
            if self.random_seed:
                random.seed(self.random_seed)

            indexes = random.sample(seg_range, piece_segs)
            indexes.sort()

            item = []
            for s_i in indexes:
                item.append(feat[s_i:s_i+piece_len])
            
            # item = []

            # for s_i in range(piece_segs):
            #     item.append(feat[s_i*step:s_i*step+piece_len])

            x.append(item)
        
        # print(labels)
        return x

    def __getitem__(self, idx):
        # print(len(self.segments[idx]), len(self.segments[idx][0]))
        segment = torch.cat(self.x[idx]).double()

        # transform
        segment = segment.permute(1, 0).contiguous()

        return segment