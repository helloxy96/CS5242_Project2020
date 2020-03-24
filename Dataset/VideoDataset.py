import torch.utils.data as tud

class VideoDataset(tud.dataset):
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


