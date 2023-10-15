import os
import torch
from model import FastSpeech2, ScheduledOptim
import constants

def get_model(restore_step, device, train=False):
    
    model = FastSpeech2(constants).to(device)
    if restore_step:
        ckpt_path = constants.CKPT_PATH +"/{}.pth.tar".format(restore_step)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(model, constants, restore_step)
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


