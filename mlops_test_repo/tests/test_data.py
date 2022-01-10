from dtumlops.data.data_utils import mnist, retrieve_mnist
from dtumlops.utils import get_project_root
import os
import pytest

mnist_n_train_samples = [25000, 40000, 60000] # three common sizes

mnist_files = os.path.join(get_project_root(),"data", "raw", "corruptmnist")
@pytest.mark.skipif(not os.listdir(mnist_files), reason="Data Mnist files not found")
def test_data_utils_mnist():
    """asserts porperties of the Mnist dataset"""
    (trainX, trainY), (testX, testY) = mnist()
    assert trainX.shape[0] == trainY.shape[0], "X_train and Y_train lenght differs"
    assert trainX.shape[0] in mnist_n_train_samples, f"not {mnist_n_train_samples} samples but {trainY.shape[0]} in the MNIST training set"
    assert testX.shape[0] == testY.shape[0], "testX and testY lenght differs"
    assert trainX.shape[1:] == testX.shape[1:], "difference in train and test shape"
    assert trainY.shape[1:] == testY.shape[1:], "difference in train and test shape"
    return

@pytest.mark.skipif(not os.path.join(get_project_root(),"data", "processed", "mnist.pt"), reason="Data Mnist processed files not found")
def test_data_utils_mnist():
    """asserts porperties of the Mnist dataset"""
    (trainX, trainY), (testX, testY) = retrieve_mnist()
    assert trainX.shape[0] == trainY.shape[0], "X_train and Y_train lenght differs"
    assert trainX.shape[0] in mnist_n_train_samples, f"not {mnist_n_train_samples} samples but {trainY.shape[0]} in the MNIST training set"
    assert testX.shape[0] == testY.shape[0], "testX and testY lenght differs"
    assert trainX.shape[1:] == testX.shape[1:], "difference in train and test shape"
    assert trainY.shape[1:] == testY.shape[1:], "difference in train and test shape"
    return
