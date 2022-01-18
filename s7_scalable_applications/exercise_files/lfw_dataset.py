"""
LFW dataloading
"""
import argparse
import time
import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import transforms
from matplotlib import pyplot as plt

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        self.data_files = glob.glob(
            os.path.join(path_to_folder, "**","*.jpg")
        )
        self.classes = list({s.split(os.path.sep)[-2] for s in self.data_files})
        self.labels = [self.classes.index(
            l.split(os.path.sep)[-2]) for l in self.data_files]

    def __len__(self):
        return len(self.data_files)  # TODO: fill out

    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        image = Image.open(self.data_files[index])
        label = np.array(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(label)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



def test_speed(path_to_folder, num_workers, visualize_batch, get_timing):
    lfw_trans = transforms.Compose(
        [transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()]
    )

    # Define dataset
    dataset = LFWDataset(path_to_folder, lfw_trans)

    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=num_workers
    )

    if visualize_batch:
        img, label = next(iter(dataloader))
        img = [img[0], img[1]]
        grid = make_grid(img)
        # show(grid)

    if get_timing:
        # lets do so repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")
    return res

if __name__ == "__main__":
   
    path_to_folder="/home/michi/.datasets/LFW"
    num_workers=2
    visualize_batch=False
    get_timing=True
    x = range(0,5)
    results = [test_speed(path_to_folder, num_workers, visualize_batch, get_timing).mean() for i in x]
    
    plt.errorbar(x, results)
    plt.show()
    print()
    