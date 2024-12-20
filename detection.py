import torch
from torch import nn


class TrafficSignDetectionCNN(nn.Module):
    def __init__(self):
        super(TrafficSignDetectionCNN, self).__init__()
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

        # Placeholder for input size
        self._initialize_classifier()

    def _initialize_classifier(self):
        # 用一个虚拟输入通过卷积部分，动态计算全连接层的输入大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 2048, 2048)  # 假设输入是2048x2048的图像
            feature_output = self.features(dummy_input)
            self.flatten_size = feature_output.view(1, -1).size(1)

        # 根据动态计算的大小定义全连接层
        self.classifier = nn.Sequential(
            # conv6
            nn.Linear(self.flatten_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # conv7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # conv8: bbox regression, pixel-wise classification, label classification
            nn.Linear(4096, 256),  # bbox regression
            nn.Linear(4096, 128),  # pixel-wise classification
            nn.Linear(4096, 1000)  # label classification
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
