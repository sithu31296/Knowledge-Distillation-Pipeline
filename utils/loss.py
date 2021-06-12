import torch
from torch import nn
from torch.nn import functional as F

class KDLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """
    def __init__(self, alpha, temp) -> None:
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.kd_loss = nn.KLDivLoss()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred_student, pred_teacher, target):
        pred_student = F.log_softmax(pred_student / self.temp, dim=1)
        pred_teacher = F.softmax(pred_teacher / self.temp, dim=1)
        loss = self.kd_loss(pred_student, pred_teacher) * (self.alpha * self.temp * self.temp)
        loss += self.entropy_loss(pred_student, target) * (1. - self.alpha)

        return loss
