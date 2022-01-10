import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.conv3 = nn.Conv2d(8, 4, 5)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError(
                f"Expected each sample to have shape [1, 28, 28] but have {x.shape}"
            )
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat1(x)
        # output so no dropout here
        x = F.log_softmax(self.fc1(x), dim=1)

        return x
