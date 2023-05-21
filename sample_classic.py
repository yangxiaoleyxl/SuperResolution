import time

from torch.utils.data import DataLoader
from torchvision import transforms

from data.datasets import CelebA_HQ_Dataset
from utils.device import device, interpolation

if __name__ == "__main__":
    # dataset and dataloader
    args = {"pre_upsample": False}
    test_dataset = CelebA_HQ_Dataset("data/test_data.txt", 64, 256, args)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model_list = ["BICUBIC", "BILINEAR", "NEAREST"]

    sample_index = 8
    for model_name in model_list:
        cnt = 0
        for (LR_img, _) in test_dataloader:
            if cnt == sample_index:
                LR_img = LR_img.to(device())
                SR_img = interpolation(LR_img[0], model_name).unsqueeze(0)
                SR_img = transforms.ToPILImage()(SR_img[0])
                SR_img.save("output" + str(sample_index) + "/" + model_name + "_SR.png")
                break
            cnt += 1

