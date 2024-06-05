import torch
from torchvision import transforms
from kornia.augmentation import Denormalize


MEAN_IMAGENET = torch.Tensor([0.485, 0.456, 0.406])
STD_IMAGENET = torch.Tensor([0.229, 0.224, 0.225])
ColorNormalization = transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
ColorDenormalization = Denormalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)

class SimCLR_Transforms:
    def __init__(self, image_size: int = 32, normalize: bool = True):
        
        transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1., 2.))], p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.1),
            transforms.RandomResizedCrop(size=(image_size, image_size))
        ]
        if normalize:
            transform += [ColorNormalization]
        self._aug = transforms.Compose(transform)
        
    def __call__(self, x: torch.Tensor):
        return transforms.Lambda(lambda x: torch.stack([self._aug(_x) for _x in x]))(x)