import torch.utils.data as tud
import torch


class VideoDataset(tud.Dataset):
    def __init__(self, videos, labels):
        super(VideoDataset, self).__init__()
        self.videos = videos
        self.lables = labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = torch.LongTensor([self.lables[idx]])
        return video, label
