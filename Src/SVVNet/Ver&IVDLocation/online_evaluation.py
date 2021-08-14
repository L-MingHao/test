# -*- encoding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from loss import  LovaszLossHinge


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = LovaszLossHinge()

    def forward(self, prediction, gt):

        loss = self.loss(prediction, gt)  # negative value

        return loss

def iou_score(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output)
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def online_evaluation(trainer):
    list_val_loss = []
    list_val_iou = []
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
            val_iou = iou_score(pred_Mask, target)
            list_val_loss.append(val_loss)
            list_val_iou.append(val_iou)

    try:
        trainer.print_log_to_file('===============================================> mean val DSC: %12.12f'
                                  % (np.mean(list_val_loss)), 'a')
        trainer.print_log_to_file('===============================================> mean val DSC: %12.12f'
                                  % (np.mean(list_val_iou)), 'a')
    except:
        pass
    # Evaluation score is the lower the better
    return np.mean(list_val_loss), np.mean(list_val_iou)
