import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import random

train = datasets.MNIST(root="data",
                       download=True,
                       train=True,
                       transform=v2.RandomResizedCrop(size=(20, 20)))
dataset = DataLoader(train, 64)

figure = plt.figure()
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train), size=(1,)).item()
    img, label = train[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.imshow(img, cmap="gray")
plt.show()


def transform_image(image):
    padding = random.randint(0, 8)
    image = v2.Pad(padding=padding)(image)
    degrees = 20
    image = v2.RandomRotation(degrees=degrees)(image)
    crop = 28
    image = v2.RandomCrop(size=(crop, crop))(image)
    noise = random.random() * 0.10
    image = image + torch.randn_like(image) * noise
    image = torch.maximum(image, torch.zeros_like(image))
    image = torch.minimum(image, torch.ones_like(image)*255)
    return image


# fig, plot = plt.subplots(3, 3)
# for j in range(3):
#     for i in range(3):
#         r = random.randint(a=0, b=len(train)-1)
#         plot[j][i].imshow(transform_image(train[r][0]).squeeze(), cmap='gray')
#         plot[j][i].set_title(train[r][1])
#
# plt.show()