# -*- encoding: utf-8 -*-
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, prediction, gt):

        loss = self.loss(prediction, gt)

        return loss
