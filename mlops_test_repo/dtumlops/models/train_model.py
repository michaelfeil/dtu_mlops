import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dtumlops.data.data_utils import CustomTensorDataset, retrieve_mnist
from dtumlops.models.model import MyAwesomeModel


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_savepath = os.path.join(
            Path(__file__).resolve().parents[2],
            "models",
            f"mnist-experiment_{int(time.time())}",
        )
        os.makedirs(self.default_savepath, exist_ok=True)

    def train(self, lr=0.001, epochs=10):
        print("Training day and night")

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        model = model.to(self.device)
        (X_train, y_train, X_test, y_test) = retrieve_mnist()

        train_dataset_normal = CustomTensorDataset(
            tensors=(X_train, y_train), transform=None
        )
        trainloader = torch.utils.data.DataLoader(train_dataset_normal, batch_size=16)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = {"l": [], "e": [], "a": []}

        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Move to GPU for acceleration
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                with torch.no_grad():
                    # images, labels = next(iter(testloader))
                    images, labels = X_test.to(self.device), y_test.to(self.device)
                    ps = torch.exp(model(images))
                    acc = torch.eq(labels.argmax(1), ps.argmax(1)).sum() / labels.size(
                        0
                    )

                print(f"Accuracy: {acc.item()}, running_loss {running_loss}")
            history["l"].append(running_loss)
            history["a"].append(acc.item())
            history["e"].append(e)
        print("Training done")
        # TODO save model
        path = os.path.join(self.default_savepath, "model_cnn.pt")
        torch.save(model.state_dict(), path)
        print(f"saved model to {path}")
        return history

    def plot_train(self, history):
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))

        ax1.plot(history["e"], history["l"], c="r", label="train loss")
        ax1.set_xlabel("epoch", fontsize=25)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel("train loss", color="r", fontsize=25)

        ax2 = ax1.twinx()
        ax2.plot(history["e"], history["a"], c="b", label="test acc")
        ax2.set_ylabel("test acc", color="lime", fontsize=25)

        ax1.legend(loc=0)
        plt.savefig(os.path.join(self.default_savepath, "training_curve.png"))

    def evaluate(self):

        print("Evaluating until hitting the ceiling")

        load_model_from = os.path.join(self.default_savepath, "model_cnn.pt")
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(load_model_from))

        (_, _, X_test, y_test) = retrieve_mnist()
        ps = model(X_test)
        ps = torch.exp(ps)
        acc = torch.eq(y_test.argmax(1), ps.argmax(1)).sum() / y_test.size(0)
        print(f"evaluted acc is: {acc}")


if __name__ == "__main__":
    te = TrainOREvaluate()
    history = te.train()
    te.plot_train(history)
    te.evaluate()
