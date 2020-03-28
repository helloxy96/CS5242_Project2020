import torch.utils.data as tud
import torch
import random


class VideoDataset(tud.Dataset):
    def __init__(self, videos, labels):
        super(VideoDataset, self).__init__()
        self.videos = videos
        self.lables = labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.lables[idx]
        return video, label


class VideoTrainDataset(tud.Dataset):
    def __init__(self, videos, labels, frames):
        super(VideoTrainDataset, self).__init__()
        self.videos = videos
        self.lables = labels
        self.frames = frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        video_len = video.shape[0]
        if video_len >= self.frames:
            start_idx = random.randint(0, video_len - self.frames)
            temp = video[start_idx: start_idx + self.frames]
            seq_length = self.frames
        elif video_len < self.frames:
            seq_length =video_len
            pad_num = self.frames - video_len
            pad_Tensor = torch.zeros((pad_num, 400), dtype=torch.float64)
            temp = torch.cat((video, pad_Tensor))
        label = self.lables[idx]
        return temp, label, seq_length
