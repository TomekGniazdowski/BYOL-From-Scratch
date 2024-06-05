import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from byol_from_scratch.transforms import ColorDenormalization


def get_model_n_layers(model: nn.Module, n: int):
    return nn.Sequential(*(list(model.children())[:n]))

def display_images(
    images: torch.Tensor, 
    labels: torch.Tensor,
    sample_random: bool = True, 
    rows: int = 5, 
    columns: int = 5, 
    size: int = 9, 
    unnorm: bool = True
    ):
    if sample_random:
        img_ids = np.random.randint(0, len(images), rows * columns)
        images = images[img_ids]
        labels = labels[img_ids]
    if unnorm:
        images = ColorDenormalization(images)
    
    _, axs = plt.subplots(rows, columns, figsize=(size, size))
    axs = axs.flatten()
    for i in range(columns * rows):
        img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        if img.dtype == np.float32:
            img = np.clip(img, 0, 1)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(labels[i])
    plt.show()