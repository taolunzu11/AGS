
#微调

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utilities import get_filename
from models import *
import config


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        使用预处理Cnn14作为子模块的新任务分类器。
        """
        super(Transfer_Cnn14, self).__init__()          #就是继承父类的init方法
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer        转移到另一个任务层
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)          #nn.Linear表示线性变换，官方文档给出的数学计算公式是y = xA^T + b

        if freeze_base:
            # Freeze AudioSet pretrained layers     冻结AudioSet预处理层
            for param in self.base.parameters():                #parameters()会返回一个生成器（迭代器）
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)        #初始化线性或卷积层

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)     #用来加载torch.save() 保存的模型文件。
        self.base.load_state_dict(checkpoint['model'])          #用于将预训练的参数权重加载到新的模型之中

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)       #log_softmax是计算损失的时候常用的一个函数，
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


def train(args):

    # Arugments & parameters        排列和参数
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path        #预处理检查点路径
    freeze_base = args.freeze_base
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    classes_num = config.classes_num


    pretrain = True if pretrained_checkpoint_path else False
    
    # Model
    Model = eval(model_type)            #返回传入字符串的表达式的结果。
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base)

    # Load pretrained model        加载预训练模型
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))        #记录info（证明事情按预期工作。）级别的日志信息
        model.load_from_pretrain(pretrained_checkpoint_path)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))           #作用：返回gpu数量。
    model = torch.nn.DataParallel(model)                #分布式训练

    if 'cuda' in device:
        model.to(device)

    print('Load pretrained model successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')         #创建子命令


    """
    创建 ArgumentParser() 对象
    调用 add_argument() 方法添加参数
    使用 parse_args() 解析添加的参数
    """


    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--sample_rate', type=int, required=True)
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)      #获取文件名

    if args.mode == 'train':
        train(args)

        #训练

    else:
        raise Exception('Error argument!')