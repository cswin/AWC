
import torch.nn as nn
from tensorflow.python.keras import backend as K

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def calc_mse_loss(item1, item2, batch_size):
    criterion = nn.MSELoss(reduce=False)
    return criterion(item1, item2).sum() / batch_size


def dice_loss_calculate(y_true, y_pred, gpu):
    smooth = 1.

    iflat = y_pred.view(-1).float().cuda(gpu)
    tflat = y_true.view(-1).float().cuda(gpu)
    intersection = (iflat * tflat).sum()
    score = 1-((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return score

def dice_loss(y_true, y_pred, gpu):

    score0 = dice_loss_calculate(y_true[:, :, :, 0], y_pred[:, :, :, 0], gpu)
    score1 = dice_loss_calculate(y_true[:, :, :, 1], y_pred[:, :, :, 1], gpu)
    score = 0.5 * score0 + 0.5 * score1

    return score



