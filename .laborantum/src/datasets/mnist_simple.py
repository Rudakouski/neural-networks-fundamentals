import os
import torch
import numpy as np
import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        root = os.path.expanduser('~/')
        dataset = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.X = np.array([np.array(dataset[i][0]) for i in range(len(dataset))], dtype=np.uint8)
        self.y = np.array([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index].astype(np.float32)
        image = (image / 127.5) - 1.0
        label = self.y[index]
        return {
            'image': torch.from_numpy(image),
            'label': torch.tensor(label, dtype=torch.long)
        }