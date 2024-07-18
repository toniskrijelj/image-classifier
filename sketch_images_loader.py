import numpy as np
import torch
import random
import urllib.request
from os.path import exists
import utils


def load(labels):
    for label in labels:
        if not exists('data/' + label + '.npy'):
            url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + label + '.npy'
            url = url.replace(' ', '%20')
            urllib.request.urlretrieve(url, 'data/' + label + '.npy')

    train = []
    test = []

    for label, i in zip(labels, range(len(labels))):
        imgs = np.load('data/' + label + '.npy')
        np.random.shuffle(imgs)

        train_len = 20000
        test_len = 1000

        for j in range(train_len):
            train.append((imgs[j], [i]))
        for j in range(test_len):
            test.append((imgs[j + train_len], [i]))

    random.shuffle(train)
    random.shuffle(test)
    train_tensor = torch.cat([torch.tensor(train[i][0]) for i in range(len(train))], dim=0)
    train_label_tensor = torch.cat([torch.tensor(train[i][1]) for i in range(len(train))], dim=0)
    test_tensor = torch.cat([torch.tensor(test[i][0]) for i in range(len(test))], dim=0)
    test_label_tensor = torch.cat([torch.tensor(test[i][1]) for i in range(len(test))], dim=0)

    batch_size = 50
    train_tensor = train_tensor.reshape((len(train) // batch_size, batch_size, 1, 28, 28))
    train_label_tensor = train_label_tensor.reshape((len(train) // batch_size, batch_size))
    test_tensor = test_tensor.reshape((len(test) // batch_size, batch_size, 1, 28, 28))
    test_label_tensor = test_label_tensor.reshape((len(test) // batch_size, batch_size))
    dataset2 = []
    testset = []

    for i in range(len(train) // batch_size):
        dataset2.append((train_tensor[i] / 255.0, train_label_tensor[i]))
    for i in range(len(test) // batch_size):
        testset.append((test_tensor[i] / 255.0, test_label_tensor[i]))

    dataset = []
    for batch in dataset2:
        x, y = batch
        x = torch.cat((x, utils.transform_images(x), utils.transform_images(x)), dim=0)
        y = torch.cat((y, y, y), dim=0)
        dataset.append((x, y))
    return dataset, testset, len(labels)
