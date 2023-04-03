#!/bin/bash
WORKSPACE=${1:-"/home/cn/lcr/pann/audioset_tagging_cnn-master/workspaces/pann-82-5-20"}   # Default argument.

python3 /home/cn/lcr/pann/audioset_tagging_cnn-master/pytorch/main.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='DaiNet19' \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=32 \
    --learning_rate=1e-4 \
    --resume_iteration=0 \
    --early_stop=20000 \
    --cuda
:<<!
# Plot statistics
python3  /home/cn/lcr/pann/audioset_tagging_cnn-master/utils/plot_statistics.py plot \
    --dataset_dir=$DATASET_DIR \
    --workspace=$WORKSPACE \
    --select=1_aug
!
