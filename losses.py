"""
Loss functions for GEMASTIK iTransformer Pipeline.
v4: DiceFocal hybrid loss + sqrt(pos_weight) + label smoothing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    POS_WEIGHTS_ORDERED, LAMBDA_FORECAST, LAMBDA_SPIKE,
    GAMMA_FOCAL, LABEL_SMOOTHING,
)


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing and sqrt(pos_weight).
    FL(p_t) = -(1 - p_t)^gamma * BCE(logits, smoothed_targets)"""

    def __init__(self, pos_weights_tensor, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.register_buffer('pos_weights', pos_weights_tensor)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    """Dice Loss for binary classification.
    Measures overlap between predicted and true positives.
    Complementary to Focal Loss -- penalizes both FP and FN equally."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=0)
        union = probs.sum(dim=0) + targets.sum(dim=0)
        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()


class DiceFocalLoss(nn.Module):
    """Hybrid Dice + Focal Loss for extreme class imbalance.
    Dice handles the overlap objective, Focal handles hard examples."""

    def __init__(self, pos_weights_tensor, gamma=2.0, label_smoothing=0.1,
                 dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(pos_weights_tensor, gamma, label_smoothing)
        self.dice  = DiceLoss()
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits, targets):
        l_focal = self.focal(logits, targets)
        l_dice  = self.dice(logits, targets)
        return self.dw * l_dice + self.fw * l_focal


class CombinedLoss(nn.Module):
    """Multi-task loss = lambda_forecast * Huber + lambda_spike * DiceFocal."""

    def __init__(self, pos_weights_tensor,
                 lambda_forecast=LAMBDA_FORECAST,
                 lambda_spike=LAMBDA_SPIKE,
                 gamma=GAMMA_FOCAL,
                 label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.forecast_loss = nn.HuberLoss(delta=1.0)
        self.spike_loss = DiceFocalLoss(
            pos_weights_tensor, gamma=gamma, label_smoothing=label_smoothing
        )
        self.lf = lambda_forecast
        self.ls = lambda_spike

    def forward(self, y_pred, y_true, spike_pred, spike_true):
        l_fc = self.forecast_loss(y_pred, y_true)
        l_sp = self.spike_loss(spike_pred, spike_true)
        total = self.lf * l_fc + self.ls * l_sp
        return total, l_fc.item(), l_sp.item()


def build_criterion(device):
    """Create combined loss with sqrt(pos_weights) on the correct device."""
    # POS_WEIGHTS_ORDERED already has sqrt applied (done in config.py)
    pos_weights_tensor = torch.tensor(POS_WEIGHTS_ORDERED, dtype=torch.float32).to(device)
    criterion = CombinedLoss(pos_weights_tensor).to(device)
    return criterion
