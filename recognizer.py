import torch
from torch import nn, optim


class RecognizerCNN(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(), optimizer=None):
        super(RecognizerCNN, self).__init__()
        self.criterion = criterion
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            # conv6
            nn.Linear(82944, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # conv7
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # conv8: bbox regression, pixel-wise classification, label classification
            nn.Linear(2048, 256),  # bbox regression
            # nn.Linear(2048, 128),  # pixel-wise classification
            # nn.Linear(2048, 256)  # label classification
        )
        if not optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

