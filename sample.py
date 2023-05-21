import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from data.datasets import CelebA_HQ_Dataset
from model_utils import initialize_model_and_optimizer
from utils.device import device

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
    # test_dataset = CelebA_HQ_Dataset("data/test_data.txt", 64, 256)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model, _ = initialize_model_and_optimizer(args)
    model.load_state_dict(torch.load(args["checkpoint_path"]))

    sample_index = 14
    cnt = 0
    # Sample
    for (LR_img, HR_img) in test_dataloader:
        if cnt == sample_index:
            LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
            with torch.no_grad():
                SR_img = model(LR_img)
                # LR_img = transforms.ToPILImage()(LR_img[0])
                SR_img = transforms.ToPILImage()(SR_img[0])
                HR_img = transforms.ToPILImage()(HR_img[0])
                # LR_img.save("output0/" + args['model'] + "_LR.png")
                SR_img.save("output" + str(sample_index) + "/" + args['model'] + "_SR.png")
                HR_img.save("output" + str(sample_index) + "/GT.png")
            break

        cnt += 1
