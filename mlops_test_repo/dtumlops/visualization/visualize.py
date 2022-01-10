import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from dtumlops.data.data_utils import retrieve_mnist
from dtumlops.models.model import MyAwesomeModel

features = {}


class VisCNNFeatures:
    def get_features(self, name):
        """HELPER FUNCTION FOR FEATURE EXTRACTION"""

        def hook(model, input, output):
            features[name] = output.detach()

        return hook

    def visualize_cnn_features(
        self,
        experiment_folder="/home/michi/dtu_mlops/s2_organisation_and_version_control/exercise_files/mlops_test_repo/models/mnist-experiment_1641392205",
    ):
        """
        load model and visualize CNN feautres with T-SNE
        
        Params:
            experiment_folder (str): folder which contrains a cnn.pth model file 
                default ".../mnist-experiment_1641392205"
        """

        load_model_from = os.path.join(experiment_folder, "model_cnn.pt")
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(load_model_from))

        model.flat1.register_forward_hook(self.get_features("activations"))

        (X_train, y_train, _, _) = retrieve_mnist()
        X_train, y_train = X_train[:5000, :, :, :], y_train[:5000, :]
        model(X_train)
        feature = features["activations"].cpu().numpy()
        label = y_train.cpu().numpy().argmax(axis=1)

        tsne = TSNE(n_components=2)
        feature_embedded = tsne.fit_transform(feature)

        sns.scatterplot(
            feature_embedded[:, 0],
            feature_embedded[:, 1],
            hue=label,
            legend="full",
            palette=sns.color_palette("bright", 10),
        )
        plt.savefig(os.path.join(experiment_folder, "feature_plot.png"))
        print("done")


if __name__ == "__main__":
    VisCNNFeatures().visualize_cnn_features()
