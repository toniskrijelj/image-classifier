import random
import torch
from torch import nn, save, load
from torch.optim import Adam


class ImageClassifier:

    class NeuralNetwork(nn.Module):
        def __init__(self, in_channels, width, height, out_features):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 32, (3, 3)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3)),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*(width-6)*(height-6), out_features)
            )

        def forward(self, x):
            return self.model(x)

    def __init__(self, device, in_channels, width, height, out_features):
        self.device = device
        self.nn = self.NeuralNetwork(in_channels, width, height, out_features).to(device)
        self.opt = Adam(self.nn.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, file_name):
        with open(file_name+'.pt', 'wb') as f:
            save(self.nn.state_dict(), f)

    def load(self, file_name):
        with open(file_name+'.pt', 'rb') as f:
            self.nn.load_state_dict(load(f, map_location=torch.device(self.device)))

    # dataset in shape: [[x, y]], x images, y labels
    # returns loss, correct, wrong, correct_test, wrong_test
    def train_loop(self, dataset, testset) -> tuple:
        random.shuffle(dataset)
        random.shuffle(testset)
        correct = 0
        wrong = 0
        total_loss = 0
        for batch in dataset:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.nn(x)
            loss = self.loss_fn(yhat, y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            yhat = torch.argmax(yhat, dim=1, keepdim=False)
            result = yhat - y
            current_wrong = torch.count_nonzero(result)
            wrong += current_wrong
            correct += result.shape[0] - current_wrong
            total_loss += loss.item()

        correct_test = 0
        wrong_test = 0
        for batch in testset:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.nn(x)

            yhat = torch.argmax(yhat, dim=1, keepdim=False)
            result = yhat - y
            current_wrong = torch.count_nonzero(result)
            wrong_test += current_wrong
            correct_test += result.shape[0] - current_wrong

        return total_loss, correct, wrong, correct_test, wrong_test

    def calculate(self, x):
        return self.nn(x)
