"""
Model 1
Structured after the Towards Data Science model
https://towardsdatascience.com/american-sign-language-hand-gesture-recognition-f1c4468fb177
"""

# Python & PyTorch imports
import torch.nn as nn # nn contains nn.Module, which should be the subclass of any networks created
import torch

class model1(nn.Module):

    def __init__(self, num_classes=1000):
        super(model1, self).__init__()

        # Conv2d args are as follows
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        # Layer 1
        # Conv2d from 1x28x28 -> 32x28x28
        # Maxpool from 32x28x28 -> 32x14x14
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2
        # Conv2d from 32x14x14 -> 64x14x14
        # Maxpool from 64x14x14 -> 64x7x7
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dropout function not included, but would be placed here otherwise
        # self.drop_out = nn.Dropout()

        # Linear args are as follows
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 26)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.layer1(x)
        out = self.layer2(out)
        # Flatten
        out = out.reshape(out.size(0), -1)
        # Dense
        out = self.fc1(out)
        out = self.fc2(out)
        return out