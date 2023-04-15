from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MINISTLableWiseInMemoDataset(Dataset):
    def __init__(self, root : str = './data', lbl : int = 0, train = True, transform = None) -> None:
        super().__init__()
        self._lbl = lbl
        self._dataset = torchvision.datasets.MNIST(root, train=train, transform=transform, download=True)
        self._data = [(img, label) for (img, label) in self._dataset if label == lbl]

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        return self._data[index]
    

if __name__ == '__main__':
    dataset = MINISTLableWiseInMemoDataset()
    print(len(dataset))
    print(dataset[0])