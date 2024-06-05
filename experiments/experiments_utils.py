import torch
from torchvision import transforms


MEAN_IMAGENET = torch.Tensor((0.485, 0.456, 0.406))
STD_IMAGENET = torch.Tensor((0.229, 0.224, 0.225))
CifarTransformsSupervised = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
])