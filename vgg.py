import torch.nn as nn

class VGG16Mod(nn.Module):
    def __init__(self, inc, out, should_maxpool):
        super(VGG16Mod, self).__init__()
        
        self.should_maxpool = should_maxpool
        
        self.layer = nn.Sequential(
            nn.Conv2d(inc, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.layer(x)
        if self.should_maxpool:
            y = self.maxpool(y)

        return y

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layers = nn.Sequential(
            VGG16Mod(3, 64, False), # Conv_1
            VGG16Mod(64, 64, True), # Conv_2
            VGG16Mod(64, 128, False), # Conv_3
            VGG16Mod(128, 128, True), # Conv_4
            VGG16Mod(128, 256, False), # Conv_5
            VGG16Mod(256, 256, False), # Conv_6
            VGG16Mod(256, 256, True), # Conv_7
            VGG16Mod(256, 512, False), # Conv_8
            VGG16Mod(512, 512, False), # Conv_9
            VGG16Mod(512, 512, True), # Conv_10
            VGG16Mod(512, 512, False), # Conv_11
            VGG16Mod(512, 512, False), # Conv_12
            VGG16Mod(512, 512, True), # Conv_13
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, num_classes))

    def forward(self, x):
        y = self.layers(x)
        y = y.reshape(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y
    