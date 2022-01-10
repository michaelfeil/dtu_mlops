import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def preload_mnist(logger, input_filepath, output_filepath):
    logger.info(
        f"making MNIST from raw data {input_filepath} to final data set {output_filepath}  "
    )
    (X_train, y_train), (X_test, y_test) = load_mnist(
        os.path.join(input_filepath, "corruptmnist")
    )
    mnist_save_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    output_path = os.path.join(output_filepath, "mnist.pt")
    torch.save(mnist_save_dict, output_path)
    logger.info(f"saving MNIST to {output_path} as final data set")


def retrieve_mnist(output_filepath=None):
    """load X_train, y_train, X_test, y_test from mnist.pt"""
    if output_filepath is None:
        output_filepath = os.path.join(
            Path(__file__).resolve().parents[2], "data", "processed"
        )

    x = torch.load(os.path.join(output_filepath, "mnist.pt"))
    return (x["X_train"], x["y_train"], x["X_test"], x["y_test"])


def mnist():
    # exchange with the corrupted mnist dataset
    # get current directory
    curr_path = Path(__file__).resolve()
    data_folder = os.path.join(
        curr_path.parent.parent.parent.parent, "data", "corruptmnist"
    )
    return load_mnist(data_folder)


def load_mnist(data_folder):
    trainX = []
    trainY = []
    testX = []
    testY = []

    for filename in os.listdir(data_folder):
        if not filename.endswith(".npz"):
            continue

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
