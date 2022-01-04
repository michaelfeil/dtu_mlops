import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path


def mnist():
    # exchange with the corrupted mnist dataset
    # get current directory
    curr_path = Path(__file__).resolve()
    data_folder = os.path.join(curr_path.parent.parent.parent.parent, "data", "corruptmnist")
    trainX = []
    trainY = []
    testX = []
    testY = []
    
    for filename in os.listdir(data_folder):
        if not filename.endswith(".npz"): continue
        
        data = np.load(os.path.join(data_folder, filename))
        if "train" in filename:
            trainX.append(data["images"])
            trainY.append(data["labels"])
        elif "test" in filename:
            testX.append(data["images"])
            testY.append(data["labels"])
    
    trainX = torch.Tensor(np.expand_dims(np.vstack(trainX), 1))
    trainY = torch.Tensor(np.concatenate(trainY))
    trainY = torch.nn.functional.one_hot(trainY.type(torch.int64))
    testX = torch.Tensor(np.expand_dims(np.vstack(testX), 1))
    testY = torch.Tensor(np.concatenate(testY))
    testY = torch.nn.functional.one_hot(testY.type(torch.LongTensor))
    
    assert trainX.shape[0] == trainY.shape[0]
    assert testX.shape[0] == testY.shape[0]
    assert trainX.shape[1:] == testX.shape[1:]

    return (trainX, trainY), (testX, testY)

class CustomTensorDataset(Dataset):
    """
    src: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index].to(torch.float)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

if __name__ == "__main__":
    (X_train, y_train), (x_test, y_test) = mnist()
     
    print("check done")
    CustomTensorDataset(tensors=(X_train, y_train))