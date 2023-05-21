import numpy as np

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def calculate_PSNR(img1, img2):
    img1, img2 = np.array(img1.to("cpu")), np.array(img2.to("cpu"))
    return compare_psnr(img1, img2, data_range=1)


def calculate_SSIM(img1, img2):
    img1, img2 = np.array(img1[0].permute(1, 2, 0).to("cpu")), np.array(img2[0].permute(1, 2, 0).to("cpu"))
    return compare_ssim(img1, img2, win_size=3, data_range=1, multichannel=True)
