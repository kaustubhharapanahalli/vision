import torch
from torch import nn
from torch.nn import functional as F


class Architecture1(nn.Module):
    def __init__(self):
        super(Architecture1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(12, 10, 1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool2(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Architecture2(nn.Module):
    def __init__(self, dropout_ratio=0.05):
        super(Architecture2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(12, 10, 1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
