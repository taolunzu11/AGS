#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-[5,6]
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast_as"
#SBATCH --output=./log_%j.txt

#set -x
## comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../../venvast/bin/activate
#export TORCH_HOME=../../pretrained_models

model=ast
dataset=audioset
# full or balanced for audioset
set=full
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  tr_data=/home/cn/lcr/ast-master/egs/audioset/data/datafiles/train-55-3.jsonn
  lrscheduler_start=10
  lrscheduler_step=5
  lrscheduler_decay=0.5
  wa_start=6
  wa_end=25
#  echo "111111"
else
  bal=bal
  lr=1e-3
  epoch=40
  tr_data=/home/cn/lcr/ast-master/egs/audioset/data/datafiles/train-55-3.json
  lrscheduler_start=2
  lrscheduler_step=1
  lrscheduler_decay=0.5
  wa_start=1
  wa_end=5
#  echo "222222"
fi
te_data=/home/cn/lcr/ast-master/egs/audioset/data/datafiles/eval-55-3.json
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=12

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_length=1024
noise=False

metrics=all
loss=BCE
warmup=True
wa=True

exp_dir=./exp/re-55-3-ast-lr${lr}-decoupe
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_ast-3.csv --n_class 6 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end}
