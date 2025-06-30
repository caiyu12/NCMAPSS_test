from torch.utils.data import Dataset
import numpy
import torch

class TrainDataset(Dataset):
    def __init__(
            self,
            sample : numpy.ndarray,
            label :numpy.ndarray,
    ):
        assert sample.shape[0] == label.shape[0], "Every training sample must have its corresponding label!"
        self.samples = torch.tensor(sample, dtype=torch.float32)
        self.labels = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)
        self.length = self.labels.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item : int):
        return self.samples[item], self.labels[item]

class TestDataset(Dataset):
    def __init__(
            self,
            sample : numpy.ndarray,
            label :numpy.ndarray,
    ):
        assert sample.shape[0] == label.shape[0], "Every testing sample must have its corresponding label!"
        self.samples = torch.tensor(sample, dtype=torch.float32)
        self.labels = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)
        self.length = self.labels.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item : int):
        return self.samples[item], self.labels[item]