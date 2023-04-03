import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config


#推论

def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    推断音频剪辑的音频标记结果。
    """

    # Arugments & parameters        排列和参数
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path          #路径
    audio_path = args.audio_path                    #音频路径
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)        #返回传入字符串的表达式的结果。
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)           #加载模型
    model.load_state_dict(checkpoint['model'])                              #将预训练的参数权重加载到新的模型之中

    # Parallel
    if 'cuda' in str(device):               #将参数转换成字符串类型
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    """
    librosa.core
    核心功能包括从磁盘加载音频、计算各种谱图表示以及各种常用的音乐分析工具。
    为了方便起见，这个子模块中的所有功能都可以直接从顶层librosa.*名称空间访问。
    librosa.core.load
    可以直接通过librosa.*来访问函数，当然也可以通过librosa.core.*来访问。
    读取音频文件
    audio_path路径
    sr采样率
    mono是否将信号转换为单声道
    
    返回：
    y ：音频时间序列
    sr ：音频的采样率
    """

    waveform = waveform[None, :]    # (1, audio_length)     1行
    waveform = move_data_to_device(waveform, device)





    # Forward
    with torch.no_grad():       #在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
        model.eval()            ##评估集
        batch_output_dict = model(waveform, None)       #批量输出字典         ？？？

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]        #剪辑信息输出
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]          #排序索引

    """
    .argsort
    将元素从小到大排列，提取其在排列前对应的index(索引)输出。
    [::-1] 
    从大到小排列
    """


    # Print audio tagging top probabilities     输出音频标记最大概率
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],              #np.array(labels)[sorted_indexes[k]]？
            clipwise_output[sorted_indexes[k]]))
        #{:.3f}保留三位小数

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels

def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    推断音频剪辑的声音事件检测结果。
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path      #检查点路径
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num        #configs 文件夹下存放训练脚本和验证脚本的yaml配置文件，文件按模型类别存放
    labels = config.labels
    frames_per_second = sample_rate // hop_size     #每秒帧数
    #在Python中/表示浮点整除法，返回浮点结果，也就是结果为浮点数;而//在Python中表示整数除法，返回大于结果的一个最大的整数，意思就是除法结果向下取整。


    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    """
    从后往前看，会从第一个以”/”开头的参数开始拼接，之前的参数全部丢弃；
    以上一种情况为先。在上一种情况确保情况下，若出现”./”开头的参数，会从”./”开头的参数的前面参数全部保留；
    """
    create_folder(os.path.dirname(fig_path))#os.path.dirname(path)  去掉文件名，返回目录

    # Model
    Model = eval(model_type)        #返回传入字符串的表达式的结果。
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)   #map_location参数是用于重定向
    model.load_state_dict(checkpoint['model'])                  #？？？？？？？？？？

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))       #gpu的数量
    model = torch.nn.DataParallel(model)                            #分布式训练

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    #读取文件，可以是wav、mp3等格式。
    #sr采样率（默认22050，但是有重采样的功能）
    # 设置为true是单通道，否则是双通道

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)#将wavform放在device中执行

    # Forward
    with torch.no_grad():       ##在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
        model.eval()            #评估集
        batch_output_dict = model(waveform, None)           #？？？？？？？？？？？？

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]      #framewise_output逐帧输出
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    """
    .argsort
    将元素从小到大排列，提取其在排列前对应的index(索引)输出。
    [::-1] 
    从大到小排列
    np.max(array1, axis=0)的意思就是：按第一个维度（即，行）对array1进行拆分，得到array1[0, :]、array1[1, :]、array1[2, :]，
    然后对array1[0, :]、array1[1, :]、array1[2, :]的对应元素进行逐位比较，并取其最大者，构成新的ndarray。
    """


    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    #framewise_output输出最大的列
    """(time_steps, top_k)"""

    # Plot result    绘制结果
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]
    #shape[-1]代表最后一个维度

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))     #两行一列共享x轴
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    #np.abs计算数组各元素的绝对值  np.log计算数组各元素的自然对数
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    """
    创建 ArgumentParser() 对象
    调用 add_argument() 方法添加参数
    使用 parse_args() 解析添加的参数
    """

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)
        #推断音频剪辑的音频标记结果。

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)
        #推断音频剪辑的声音事件检测结果。

    else:
        raise Exception('Error argument!')