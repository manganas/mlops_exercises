"""
LFW dataloading
"""
import argparse
import time
from pathlib import Path

from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from tqdm import tqdm

# also with:
# from torchvision.datasets import ImageFolder


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        self.image_paths = list(Path(path_to_folder).glob("**/*.jpg"))
        # print(len(self.image_paths)) ## Sanity check

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.image_paths[index])
        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="lfw-deepfunneled/", type=str)
    parser.add_argument("-batch_size", default=16, type=int)
    parser.add_argument("-num_workers", default=2, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose(
        [
            transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5) * 3, (1) * 3),
        ]
    )

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    plt.rcParams["savefig.bbox"] = "tight"

    def show(imgs):
        if not isinstance(imgs, List):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = F.to_pil_image(img.detach())
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(
            "grid.jpg",
        )
        plt.show()

    if args.visualize_batch:
        imgs = next(iter(dataloader))
        grid = make_grid(imgs)
        show(grid)

    if args.get_timing:
        # lets do some repetitions
        res = []
        for _ in tqdm(range(5), desc="Iteration"):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")
