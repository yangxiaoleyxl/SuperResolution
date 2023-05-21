import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import CelebA_HQ_Dataset
from utils.device import device, interpolation
from utils.logger import AverageMeter
from utils.metrics import calculate_PSNR, calculate_SSIM

if __name__ == "__main__":
    # dataset and dataloader
    args = {"pre_upsample": False}
    test_dataset = CelebA_HQ_Dataset("data/test_data.txt", 64, 256, args)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model_list = ["BICUBIC", "BILINEAR", "NEAREST"]

    for model_name in model_list:
        # Metrics
        # psnr = AverageMeter()
        # ssim = AverageMeter()
        # with tqdm(total=len(test_dataset) - len(test_dataset) % 1) as t:
        #     t.set_description(model_name)
        #     for (LR_img, HR_img) in test_dataloader:
        #         LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
        #         SR_img = interpolation(LR_img[0], model_name).unsqueeze(0)
        #         psnr.update(calculate_PSNR(HR_img, SR_img), len(LR_img))
        #         ssim.update(calculate_SSIM(HR_img, SR_img), len(LR_img))
        #         t.update(len(LR_img))
        #
        # print('Test PSNR: {:.4f}'.format(psnr.avg))
        # print('Test SSIM: {:.4f}'.format(ssim.avg))

        # Running Time
        start_time = time.time()
        for (LR_img, HR_img) in test_dataloader:
            LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
            SR_img = interpolation(LR_img[0], model_name).unsqueeze(0)
        end_time = time.time()
        print("Speed: {:.4f} ms/img".format((end_time - start_time) / 3000 * 1000))
