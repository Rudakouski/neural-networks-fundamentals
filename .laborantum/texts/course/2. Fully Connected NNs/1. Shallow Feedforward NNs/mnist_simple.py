# src/datasets/mnist_simple.py

import os
import torch
import torchvision.datasets as datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        root = os.path.expanduser("~/data")
        os.makedirs(root, exist_ok=True)
        mnist = datasets.MNIST(root=root, train=train, download=True, transform=None)
        self.X = mnist.data   
        self.y = mnist.targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index].to(torch.float32) / 255.0 * 2.0 - 1.0
        image = image.unsqueeze(0)  
        label = self.y[index].long()
        return {'image': image, 'label': label}