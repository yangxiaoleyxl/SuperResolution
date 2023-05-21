import argparse
import copy

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import CelebA_HQ_Dataset
from model_utils import initialize_model_and_optimizer, save_best_model
from utils.device import device
from utils.logger import AverageMeter
from utils.loss import loss_fn
from utils.metrics import calculate_PSNR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = vars(parser.parse_args())

    # read config
    with open("config/" + args["model"] + ".yml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args.update(cfg)

    # dataset and dataloader
    train_dataset = CelebA_HQ_Dataset("data/train_data.txt", 64, 256, args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_dataset = CelebA_HQ_Dataset("data/val_data.txt", 64, 256, args)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args["batch_size"])

    # model and optimizer definition
    model, optimizer = initialize_model_and_optimizer(args)

    # loss function
    criterion = loss_fn(args["loss_fn"])

    # start train
    best_psnr = 0.0
    for epoch in range(args["max_epoch"]):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args["batch_size"])) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args["max_epoch"] - 1))
            for (LR_img, HR_img) in train_dataloader:
                LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
                SR_img = model(LR_img)
                loss = criterion(SR_img, HR_img)
                epoch_losses.update(loss.item(), len(LR_img))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(LR_img))

            # validation
            model.eval()
            epoch_psnr = AverageMeter()
            for (LR_img, HR_img) in val_dataloader:
                LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
                with torch.no_grad():
                    SR_img = model(LR_img).clamp(0, 255)

                epoch_psnr.update(calculate_PSNR(SR_img, HR_img), len(LR_img))
            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

            if epoch_psnr.avg > best_psnr:
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())
                save_best_model(args["model"], best_weights)
                print('best epoch: {}, psnr: {:.2f}'.format(epoch, best_psnr))
