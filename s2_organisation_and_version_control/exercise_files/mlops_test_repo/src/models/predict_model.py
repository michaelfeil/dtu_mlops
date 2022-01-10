from train_model import TrainOREvaluate

if __name__ == "__main__":
    te = TrainOREvaluate()
    history = te.train()
    te.evaluate()