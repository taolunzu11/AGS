# AGS

可以从**连接**下载我们的数据集其中包括我们的音频文件`/audioset`.`AGS_label.xlsx`是我们的标注文件,`New_label.xlsx`是我们新增加的标签对应的音频名称。`Charades_v1_test.csv`和`Charades_v1_train.csv`分别是Action Genome数据集的训练集的视频描述和测试集的视频描述，`video_label.txt`这个文件是Action Genome数据集对应的视频的类别以及视频训练集测试集的划分。`object_bbox_and_relationship_filtersmall.pkl`是Action Genome数据集人物与对象之间的关系表示。



## PANNs

### Environments

The codebase is developed with Python 3.7. Install requirements as follows:

```
cd pann
pip install -r requirements.txt
```


### 1. 数据处理

将下载好的数据集的按照训练集与测试集的比例放在`/pann/audioset`中，例如`/pann/audioset/eval-55-3`或`/pann/audioset/train-55-3`。将训练集与测试集的表格以及他们的类别和编号放在`/pann/metadata`下（可以参考`eval-82-5-1.csv`与`train-82-5-1.csv`的格式以及`class_labels_pann-5-1.csv`的格式）。

### 2. Pack waveforms into hdf5 files

The [/pann/scripts/2_pack_waveforms_to_hdf5s.sh](/pann/scripts/2_pack_waveforms_to_hdf5s.sh) script is used to pack all the raw waveforms into hdf5 files to speed up training.
```
cd /pann/scripts
bash 2_pack_waveforms_to_hdf5s.sh
```

### 3. Create training indexes

The [/pann/scripts/3_create_training_indexes.sh](/pann/scripts/3_create_training_indexes.sh) is used for creating training indexes. Those indexes are used for sampling mini-batches.

```
cd /pann/scripts
bash 3_create_training_indexes.sh
```


###  4. Train

The [/pann/scripts/4_train.sh](/pann/scripts/4_train.sh) script contains training, saving checkpoints, and evaluation.
```
cd /pann/scripts
bash 4_train.sh
```


## AST

### Environments

The codebase is developed with Python 3.7， create a virtual environment and install the dependencies.
```
cd ast
pip install -r requirements.txt
```

### 1.数据处理
需要准备数据的json文件（即`/ast/egs/audioset/data/datafiles/eval-55-3.json`和`/ast/egs/audioset/data/datafiles/train-55-3.json`）将其放在`/ast/egs/audioset/data/datafiles`下。还需要生成类别的csv文件，例如`/ast/egs/audioset/data/class_labels_ast-3.csv`。

### 2. Obtain the sampling weight file.

Once you have the json files,you will need to generate the sampling weight file of your training data.

```
cd ast/egs/audioset
python gen_weight_file.py
```
### 3. Train

```
cd ast/egs/audioset
bash run.sh
```



### Results

The computing environment is that Linux server with two NVIDIA Geforce RTX 3090, 125GB memory.

The experimental results of 4 experimental methods on the AGS dataset are shown in the following figure: 

<img src="result-acc.pdf">
<img src="result-map.pdf">

选取1,406个清晰声音数据进行实验，获得以下的实验结果：



![result](D:\新建文件夹\result.png)

