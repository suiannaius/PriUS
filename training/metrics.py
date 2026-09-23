import torch
import math
import numpy as np
import torch.nn.functional as F
from utilities import binary


def hd_score(y_true, y_pred, eps_max=100, eps=1e-8, voxelspacing=None):
    if (y_true.sum()==0) and (y_pred.sum()==0):
        hd = eps
    elif (y_true.sum()!=0) and (y_pred.sum()==0):
        hd = eps_max
    elif (y_true.sum()==0) and (y_pred.sum()!=0):
        hd = eps_max
    else:
        hd = binary.hd95(y_true, y_pred, voxelspacing=voxelspacing)
    return hd

def soft_hd95(y_true, y_pred, voxelspacing=None, num_classes=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    hd = np.zeros(num_classes - 1)
    
    if voxelspacing is not None:
        assert voxelspacing.shape[0]==y_true.shape[0] and voxelspacing.shape[1]==2, f"Check the dimension of spacing."
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t, voxelspacing=voxelspacing[i])
    else:
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t)
    return hd / y_true.shape[0]

def SDice(y_true, y_pred, epsilon, device=torch.device('cuda')):
    y_true = y_true.to(device) # [NHWD, 1]
    y_pred = y_pred.to(device) # [NHWD, 1]
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice_coef = 2 * intersection / (union + epsilon) # [1]
    return dice_coef

def calculate_dice(y_true, y_pred, epsilon, device=torch.device('cuda'), num_classes=None):
    y_true_hard = torch.argmax(y_true, axis=1, keepdims=True)
    y_pred_hard = torch.argmax(y_pred, axis=1, keepdims=True)
    dice = torch.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        dice[i] = SDice((y_pred_hard == i + 1).float(), (y_true_hard == i + 1).float(), epsilon, device) # [NHW, 1], [NHW, 1]
    return dice

def calculate_dice_from_labels(
    pred_labels,   # [N, D, H, W] int
    gt_labels,     # [N, D, H, W] int
    num_classes,
    epsilon=1e-5,
    device=torch.device("cuda")):
    dice = torch.zeros(num_classes - 1, device=device)

    pred_labels = pred_labels.to(device)
    gt_labels = gt_labels.to(device)

    for i in range(1, num_classes):
        pred_i = (pred_labels == i).float()
        gt_i = (gt_labels == i).float()

        intersection = torch.sum(pred_i * gt_i)
        union = torch.sum(pred_i) + torch.sum(gt_i)

        dice[i - 1] = 2.0 * intersection / (union + epsilon)

    return dice

def Uentropy(logits, c):
    pred = F.softmax(logits, dim=1)  # 1 4 240 240 155
    logits = F.log_softmax(logits, dim=1)  # 1 4 240 240 155
    u_all = -pred * logits / math.log(c)
    NU = torch.sum(u_all[:, 1:u_all.shape[1], :, :], dim=1)
    return NU

def soft_hd95(y_true, y_pred, voxelspacing=None, num_classes=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    hd = np.zeros(num_classes - 1)
    
    if voxelspacing is not None:
        assert voxelspacing.shape[0]==y_true.shape[0] and voxelspacing.shape[1] in (2, 3), f"Check the dimension of spacing."
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t, voxelspacing=voxelspacing[i])
    else:
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t)
    return hd / y_true.shape[0]
