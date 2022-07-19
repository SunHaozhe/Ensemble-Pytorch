import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import sklearn

from torchensemble import GradientBoostingClassifier
from torchensemble.utils.logging import set_logger

# Define Your Base Estimator


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, data):
        data = data.view(data.size(0), -1)
        output = F.relu(self.linear1(data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train = datasets.MNIST('../Dataset', train=True,
                       download=True, transform=transform)
test = datasets.MNIST('../Dataset', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

# Set the Logger
logger = set_logger('classification_mnist_mlp')

# Define the ensemble
model = GradientBoostingClassifier(
    estimator=MLP,
    n_estimators=3,
    cuda=False,
    shrinkage_rate=1
)

# compute class weights
classes = []
for X, y in train_loader.dataset:
    classes.append(y)
class_weights = sklearn.utils.class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(classes), y=np.asarray(classes))
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights.to("cpu")

# Set the criterion
criterion = nn.CrossEntropyLoss(weight=class_weights)
print(id(criterion))
model.set_criterion(criterion)

# Set the optimizer
model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

# Train and Evaluate
model.fit(
    train_loader,
    epochs=3,
    test_loader=test_loader,
)
