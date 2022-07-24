"""
Code adapted from:
https://ensemble-pytorch.readthedocs.io/en/latest/quick_start.html#example-on-mnist
https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/examples/classification_cifar10_cnn.py 
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from torchensemble import BaggingClassifier
from torchensemble.utils.logging import set_logger

# Define Your Base Estimator


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


train_transformer = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

test_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

train = datasets.CIFAR10('../tmp_cifar10', train=True,
                         download=True, transform=train_transformer)
test = datasets.CIFAR10('../tmp_cifar10', train=False,
                        transform=test_transformer)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

# Set the Logger
logger = set_logger('bagging_cifar10_lenet5')

# Define the ensemble
model = BaggingClassifier(
    estimator=LeNet5,
    n_estimators=5,
    cuda=True,
    n_jobs=1,
)

# Set the criterion
criterion = nn.CrossEntropyLoss()
model.set_criterion(criterion)

# Set the optimizer
model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

# Train and Evaluate
model.fit(
    train_loader,
    epochs=100,
    test_loader=test_loader,
)
