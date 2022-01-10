from dtumlops.models.train_model import TrainOREvaluate
import pytest

def test_train():
    tr_or_ev = TrainOREvaluate()
    history = tr_or_ev.train(epochs=2)
    train_loss = history["l"]
    assert train_loss[0] > train_loss[1], "loss not decreasing from epoch 0 to epoch 1"
    
    tr_or_ev.plot_train(history)
    res = tr_or_ev.evaluate()
    assert res is None
    