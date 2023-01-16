import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, resnet50

import timeit
import torch.utils.benchmark as benchmark

# Parameters
batch_size = 1
n_workers = 4
epochs = 1
first_n_el = 30

# benchmark
tries = 5


# Load cifar dataset
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers
)


cifar_10_classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

weights = ResNet50_Weights.DEFAULT
transforms = weights.transforms()
resnet = resnet50(weights=weights)

resnet.eval()


def predict_bare():
    for epoch in range(epochs):
        for img, target in trainloader:
            img = transforms(img)
            preds = resnet(img)
            break
        break
    return


resnet_scripted = torch.jit.script(resnet)


def predict_scripted():
    for epoch in range(epochs):
        for img, target in trainloader:
            img = transforms(img)
            preds = resnet_scripted(img)
            break
        break
    return


# timeit
t0 = timeit.Timer(
    stmt="predict_bare()", setup="from __main__ import predict_bare", globals={}
)

t1 = timeit.Timer(
    stmt="predict_scripted()", setup="from __main__ import predict_scripted", globals={}
)

t0_ = t0.timeit(tries) / tries
t1_ = t1.timeit(tries) / tries

print(f"timeit Unscripted:  {t0_ * 1e6:>5.1f} us")
print(f"timeit Scripted:      {t1_ * 1e6:>5.1f} us")
print(f"timeit Difference: {(t1_-t0_)/t0_*100:>5.5f}%")

# torch benchmark

t0 = benchmark.Timer(
    stmt="predict_bare()", setup="from __main__ import predict_bare", globals={}
)

t1 = benchmark.Timer(
    stmt="predict_scripted()", setup="from __main__ import predict_scripted", globals={}
)

t0_ = t0.timeit(tries)
t1_ = t1.timeit(tries)

import numpy as np

print(f"torch Unscripted:  {t0_}")
print(f"torch Scripted:      {t1_}")
print(f"torch Difference: {(t1_.raw_times[0]-t0_.raw_times[0])/t0_.raw_times[0]*100}%")
