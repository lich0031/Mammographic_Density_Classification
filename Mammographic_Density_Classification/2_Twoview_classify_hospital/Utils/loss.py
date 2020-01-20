from torch import nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class Dice_Loss(nn.Module):
    def __init__(self, smooth=1.0, size_average=True):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        self.size_average = size_average
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        inputs_obj = inputs[:, 1, :, :]
        diceloss = 0.
        for input_, target_ in zip(inputs_obj, targets):
            iflat = input_.view(-1).float()
            tflat = target_.view(-1).float()
            intersection = (iflat * tflat).sum()
            loss = 1.0 - (((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)))
            diceloss += loss
        if self.size_average:
            dice_patch_loss = diceloss / inputs.shape[0]
        else:
            dice_patch_loss = diceloss
        return dice_patch_loss

class CE_Dice_Loss(nn.Module):
    def __init__(self, weight):
        super(CE_Dice_Loss, self).__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss2d()
        self.dice = Dice_Loss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.weight * ce_loss + dice_loss
        # return ce_loss + self.weight * dice_loss

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

class AT_loss(nn.Module):
    def __init__(self, size_average=True):
        super(AT_loss, self).__init__()
        self.size_average = size_average

    def forward(self, feat1, feat2):
        if self.size_average:
            return (at(feat1) - at(feat2)).pow(2).mean()
        else:
            return (at(feat2) - at(feat2)).pow(2).sum()

def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

def dice_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj>=threshold] = 1
    inputs_obj[inputs_obj<threshold] = 0
    dice = 0.
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
        dice += dice_single
    return dice

def TP_TN_FP_FN(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj>=threshold] = 1
    inputs_obj[inputs_obj<threshold] = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        TP = (iflat * tflat).sum()
        TN = ((1-iflat) * (1-tflat)).sum()
        FP = (iflat * (1-tflat)).sum()
        FN = ((1-iflat) * tflat).sum()
    return TP, TN, FP, FN


def IoU_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj >= threshold] = 1
    inputs_obj[inputs_obj < threshold] = 0
    IoU = 0.
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        IoU_single = intersection / (iflat.sum() + tflat.sum() - intersection)
        IoU += IoU_single
    return IoU