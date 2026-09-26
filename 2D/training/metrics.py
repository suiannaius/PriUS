import torch
import numpy as np
from utilities import binary


def hd_score(y_true, y_pred, eps_max=100, eps=1e-8, voxelspacing=None):
    if (y_true.sum() == 0) and (y_pred.sum() == 0):
        hd = eps
    elif (y_true.sum() != 0) and (y_pred.sum() == 0):
        hd = eps_max
    elif (y_true.sum() == 0) and (y_pred.sum() != 0):
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
        assert voxelspacing.shape[0] == y_true.shape[0] and voxelspacing.shape[1] in (
            2,
            3,
        ), f"Check the dimension of spacing."
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


def SDice(y_true, y_pred, epsilon, device=torch.device("cuda")):
    y_true = y_true.to(device)  # [NHWD, 1]
    y_pred = y_pred.to(device)  # [NHWD, 1]
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice_coef = 2 * intersection / (union + epsilon)  # [1]
    return dice_coef


def calculate_dice(
    y_true, y_pred, epsilon, device=torch.device("cuda"), num_classes=None
):
    y_true_hard = torch.argmax(y_true, axis=1, keepdims=True)
    y_pred_hard = torch.argmax(y_pred, axis=1, keepdims=True)
    dice = torch.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        dice[i] = SDice(
            (y_pred_hard == i + 1).float(),
            (y_true_hard == i + 1).float(),
            epsilon,
            device,
        )  # [NHW, 1], [NHW, 1]
    return dice
