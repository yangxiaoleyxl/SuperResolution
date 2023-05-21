import torch
from PIL import Image
from torchvision import transforms


def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def interpolation(img, method):
    img = transforms.ToPILImage()(img)
    if method == "BICUBIC":
        img = img.resize((256, 256), Image.BICUBIC)
    elif method == "BILINEAR":
        img = img.resize((256, 256), Image.BILINEAR)
    elif method == "NEAREST":
        img = img.resize((256, 256), Image.NEAREST)

    return transforms.ToTensor()(img)
