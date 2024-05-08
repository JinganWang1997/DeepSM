from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        loss = ((outputs - targets)**2).mean()

        return loss