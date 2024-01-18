import torch.nn as nn


class LeNet_300_100(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet_300_100, self).__init__()

        self.ip1 = nn.Linear(28*28, 300)
        self.relu_ip1 = nn.ReLU(inplace=True)

        self.ip2 = nn.Linear(300, 100)
        self.relu_ip2 = nn.ReLU(inplace=True)

        self.ip3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)

        x = self.ip2(x)
        x = self.relu_ip2(x)

        x = self.ip3(x)
        return x
