from torchvision.models import ResNet50_Weights, resnet50
import torch

weights = ResNet50_Weights
model = resnet50(weights=weights)
script_model = torch.jit.script(model)
script_model.save("deployable_model.pt")
