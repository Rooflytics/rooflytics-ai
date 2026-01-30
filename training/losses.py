import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE part
        bce_loss = self.bce(preds, targets)

        # Dice part
        probs = torch.sigmoid(preds)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        dice_loss = 1 - dice

        return bce_loss + dice_loss
