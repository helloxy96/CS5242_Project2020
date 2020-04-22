import torch.utils.data as tud
import torch
import math
import random
import copy


class TestDataset(tud.Dataset):
    def __init__(self, raw_data, seg_info_path = '', seed = None, chop_num=0):
        super(TestDataset, self).__init__()
        
        self.raw_data = raw_data
        self.seg_info_path = seg_info_path

        self.random_seed = seed
        self.chop_num = chop_num

        if len(seg_info_path) > 0:
            self.x = self.get_file_x(raw_data, seg_info_path)
        else:
            self.x = raw_data
        self.x = self.process(self.x)

        if chop_num > 0:
            self.x = self.chop_to_segs(self.x)
    

    def __len__(self):
        return len(self.x)

    def chop_to_segs(self, files):
        x = []
        labels = []
        chop_num = self.chop_num

        for i, file in enumerate(files):
            old_len = len(x)
            if len(file) < chop_num:
                x.extend(file)
            else:
                if chop_num == 1:
                    x.extend(file)
                else:
                    x.extend(file[:chop_num//2])
                    for j in range(0, len(file)-chop_num+1):         
                        x.append(file[j:j+chop_num])
                    x.extend(file[-(chop_num//2):])

        return x

    def get_file_x(self, raw_data, seg_info_path):
        x = []        
        f = open(seg_info_path, 'r')

        for f_i, line in enumerate(f):
            f_x = []
            if f_i >= len(raw_data):
                break

            indexes = line.split()

            for i in range(len(indexes)-1):
                begin_index, end_index = int(indexes[i]), int(indexes[i+1])
                f_x.append(raw_data[f_i][begin_index:end_index])
            
            x.append(f_x)
        return x

    def process(self, files):

        # files.shape: file_nums, seg nums in each file, 400
        # file_labels.shape: file_nums, seg nums in each file
        x = []
        labels = []

        for i, file in enumerate(files):
            f_x = []
            f_labels = []

            for j, seg in enumerate(file):

                feats = seg


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

            if len(f_x) > 0:
                x.append(f_x)
        
        # print(labels)
        return x

    def __getitem__(self, idx):
        # print(len(self.segments[idx]), len(self.segments[idx][0]))
        file = torch.tensor(self.x[idx]).double()

        if len(file.shape) == 3:
            file = file.unsqueeze(dim=0)

        # transform
        # file would be segs, 400, 50
        try:
            file = file.squeeze(dim = 2)
            file = file.permute(0, 2, 1).contiguous()
        except:
            print('exception file idx', idx)
        # labels: segs

        return file