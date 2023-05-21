import os.path

import torch
from torch import optim

from models.BSRN import BSRN
from models.DRRN import DRRN
from models.ESPCN import ESPCN
from models.FSRCNN import FSRCNN
from models.MAFFSRN import MAFFSRN
from models.RFDN import RFDN
from models.SRCNN import SRCNN
from models.architecture import IMDN_AS
from models import architecture

from utils.device import device

model_list = ["SRCNN", "FSRCNN", "ESPCN", "DRRN", "RFDN", "MAFFSRN", "BSRN", "IMDN"]
checkpoint_root = "checkpoint"


def initialize_model_and_optimizer(args):
    assert (args["model"] in model_list)
    if args["model"] == "SRCNN":
        model = SRCNN(args=args)
        optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': args["learning_rate"] * 0.1}
        ], lr=args["learning_rate"])
    elif args["model"] == "FSRCNN":
        model = FSRCNN(args=args)
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': args["learning_rate"] * 0.1}
        ], lr=args["learning_rate"])
    elif args['model'] == "ESPCN":
        model = ESPCN(args=args)
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': args["learning_rate"] * 0.1}
        ], lr=args["learning_rate"])
    elif args['model'] == "DRRN":
        model = DRRN(args=args)
        optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"], momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif args['model'] == "RFDN":
        model = RFDN()
        optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args['learning_rate'])
    elif args['model'] == "MAFFSRN":
        model = MAFFSRN(args)
        optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args['learning_rate'])
    elif args['model'] == "BSRN":
        model = BSRN()
        optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args['learning_rate'])

    elif args['model'] == "IMDN":
        # model = IMDN_AS(upscale=args["scale"])
        model = architecture.IMDN_AS( )
        optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])

    if args["use_pretrained"]:
        model.load_state_dict(torch.load(args["pretrained_path"]))

    if args["use_checkpoint"]:
        model.load_state_dict(torch.load(args["checkpoint_path"]))

    return model.to(device()), optimizer


def save_best_model(model_name, best_weights):
    torch.save(best_weights, os.path.join(checkpoint_root, model_name + ".pth"))
