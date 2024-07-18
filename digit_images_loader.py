import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import utils


def load():
    train_data = datasets.MNIST(root="data", download=True, train=True, transform=v2.ToTensor())
    test_data = datasets.MNIST(root="data", download=True, train=False, transform=v2.ToTensor())
    train_data = DataLoader(train_data, 60, shuffle=True)
    testset = DataLoader(test_data, len(test_data))

    dataset = []
    for batch in train_data:
        x, y = batch
        x = torch.cat((x, utils.transform_images(x), utils.transform_images(x), utils.transform_images(x)), dim=0)
        y = torch.cat((y, y, y, y), dim=0)
        dataset.append((x, y))

    return dataset, testset, 10
