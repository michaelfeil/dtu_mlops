import os

import pytest

from dtumlops.models.train_model import TrainOREvaluate
from dtumlops.utils import get_project_root

mnist_files = os.path.join(get_project_root(), "data", "raw", "corruptmnist")
if os.path.exists(mnist_files):
    mnist_files = os.listdir(mnist_files)
else:
    mnist_files = False


@pytest.mark.skipif(not mnist_files, reason="Data Mnist files not found")
def test_train():
    tr_or_ev = TrainOREvaluate()
    history = tr_or_ev.train(epochs=2)
    train_loss = history["l"]
    assert train_loss[0] > train_loss[1], "loss not decreasing from epoch 0 to epoch 1"

    # tr_or_ev.plot_train(history)
    res = tr_or_ev.evaluate()
    assert res is None
