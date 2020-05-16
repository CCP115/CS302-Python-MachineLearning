"""
Model 2
Structured after the Towards Data Science model
https://towardsdatascience.com/sign-language-recognition-in-pytorch-5d72688f98b7
"""

# Python & PyTorch imports
import torch.nn as nn # nn contains nn.Module, which should be the subclass of any networks created
import torch

class model2(nn.Module):

    def __init__(self, num_classes=1000):
        super(model2, self).__init__()

        # Conv2d args are as follows
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        # Conv Layer 1
        # Conv2d from 1x28x28 -> 10x26x26
        # Maxpool from 10x26x26 -> 10x13x13
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Layer 2
        # Conv2d from 10x13x13 -> 20x11x11
        # Maxpool from 20x11x11 -> 20x5x5
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Layer 3
        # Conv2d from 20x5x5 -> 30x3x3
        # Dropout from 30x3x3 -> 30x3x3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=0),
            nn.Dropout()
        )

        # Linear args are as follows
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(30 * 3 * 3, 26)
        self.act = nn.LogSoftmax()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # Conv layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Flatten
        out = out.reshape(out.size(0), -1)
        # Dense
        out = self.fc1(out)
        # Activation function
        out = self.act(out)
        return out