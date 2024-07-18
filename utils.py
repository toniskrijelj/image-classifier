import torch
from torchvision.transforms import v2
import random


def transform_images(images):
    padding = random.randint(0, 8)
    images = v2.Pad(padding=padding)(images)
    degrees = 20
    images = v2.RandomRotation(degrees=degrees)(images)
    crop = 28
    images = v2.RandomCrop(size=(crop, crop))(images)
    noise = random.random() * 0.10
    images = images + torch.randn_like(images) * noise
    images = torch.maximum(images, torch.zeros_like(images))
    images = torch.minimum(images, torch.ones_like(images)*255)
    return images
