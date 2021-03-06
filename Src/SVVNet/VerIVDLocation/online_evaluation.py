# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from Loss.SegLoss.DiceLoss import SoftDiceLoss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        #self.loss = nn.L1Loss()
        self.loss = nn.MSELoss()

    def forward(self, prediction, gt):

        loss = self.loss(prediction, gt[0])

        return loss



def online_evaluation(trainer):

    list_val_loss = []
    val_loss_function = Loss()

    with torch.no_grad():
        trainer.setting.network.eval()

        for batch_idx, case in tqdm(enumerate(trainer.setting.val_loader)):
            input_ = case[0].to(trainer.setting.device)  # tensor: (batch_size, C, D, H, W)
            target = case[1:]
            for target_i in range(len(target)):
                target[target_i] = target[target_i].to(trainer.setting.device)

            pred_Mask = trainer.setting.network(input_)
            val_loss = val_loss_function(pred_Mask, target).cpu().numpy()
            list_val_loss.append(val_loss)

    try:
        trainer.print_log_to_file('===============================================> mean val MSE: %12.12f'
                                  % (np.mean(list_val_loss)), 'a')
    except:
        pass
    # Evaluation score is the lower the better
    return np.mean(list_val_loss)
