import torch
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    二进制交叉熵损失。
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])
    #测量目标和输出之间二进制交叉熵的函数


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce