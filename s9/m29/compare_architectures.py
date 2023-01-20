import time
from torchvision.models import (
    EfficientNet_B5_Weights,
    ResNet101_Weights,
    ViT_B_16_Weights,
)

import torch
from torchvision.models import resnet101, efficientnet_b5, vit_b_16
from tqdm import tqdm

# EfficientNet_B5_Weights.IMAGENET1K_V1,
# ResNet101_Weights.IMAGENET1K_V1,
# ViT_B_16_Weights.IMAGENET1K_V1

# resnet_weights = ResNet101_Weights.IMAGENET1K_V1
resnet = resnet101()

efficientnet = efficientnet_b5()
vitnet = vit_b_16()

input = torch.randn(16, 3, 256, 256)

models = [resnet, efficientnet, vitnet]

n_reps = 5

for i, m in tqdm(enumerate(models), desc="Model"):
    tic = time.time()
    for _ in tqdm(range(n_reps), desc="Reps", leave=False):
        _ = m(input)
    toc = time.time()
    print(f"Model {i} took: {(toc - tic) / n_reps}")
