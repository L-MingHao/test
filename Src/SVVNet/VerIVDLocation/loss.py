# -*- encoding: utf-8 -*-
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        #self.loss = nn.L1Loss()
        self.loss = nn.MSELoss()

    def forward(self, prediction, gt):

        loss = self.loss(prediction, gt[0])

        return loss

