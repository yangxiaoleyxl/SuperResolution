import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import CelebA_HQ_Dataset
from model_utils import initialize_model_and_optimizer
from utils.device import device
from utils.logger import AverageMeter
from utils.metrics import calculate_PSNR, calculate_SSIM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = vars(parser.parse_args())

    # read config
    with open("config/" + args["model"] + ".yml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args.update(cfg)

    # dataset and dataloader
    test_dataset = CelebA_HQ_Dataset("data/test_data.txt", 64, 256, args)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model, _ = initialize_model_and_optimizer(args)
    model.load_state_dict(torch.load(args["checkpoint_path"]))
    model.eval()

    # Metrics
    psnr = AverageMeter()
    ssim = AverageMeter()
    with tqdm(total=len(test_dataset) - len(test_dataset) % 1) as t:
        t.set_description(args['model'])
        for (LR_img, HR_img) in test_dataloader:
            LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
            with torch.no_grad():
                SR_img = model(LR_img)
                # SR_img = SR_img.clamp(0, 255)

            psnr.update(calculate_PSNR(HR_img, SR_img), len(LR_img))
            ssim.update(calculate_SSIM(SR_img, HR_img), len(LR_img))
            t.update(len(LR_img))

    print('Test PSNR: {:.4f}'.format(psnr.avg))
    print('Test SSIM: {:.4f}'.format(ssim.avg))

    # Running Time
    # start_time = time.time()
    # for (LR_img, HR_img) in test_dataloader:
    #     LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
    #     with torch.no_grad():
    #         SR_img = model(LR_img)
    # end_time = time.time()
    # print("Speed: {:.4f} ms/img".format((end_time - start_time) / 3000 * 1000))
