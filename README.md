# AGS

You can download our dataset from https://drive.google.com/drive/folders/1kkLMdN6-RCqpYmkXH8ktwePyprf363-f?usp=share_link, which includes our audio files `/audioset`. `AGS_label.xlsx` is our annotation file, `New_label.xlsx` is the audio names corresponding to the newly added labels, and `overlap_label.xlsx` is the annotation for overlapping sound labels. `Charades_v1_test.csv` and `Charades_v1_train.csv` are the video descriptions for the training and testing sets of the Action Genome dataset, respectively. `video_label.txt` contains the category of the Action Genome dataset videos and their division into training and testing sets. `object_bbox_and_relationship_filtersmall.pkl` represents the relationship between people and objects in the Action Genome dataset.

## PANNs

### Environments

The codebase is developed with Python 3.7. Install requirements as follows:

```
cd pann
pip install -r requirements.txt
```


### 1. Data processing

Put the downloaded dataset into `/pann/audioset` according to the ratio of training set and test set, such as `/pann/audioset/eval-55-3` or `/pann/audioset/train-55-3`. Put the tables of the training set and test set, as well as their categories and numbers, in `/pann/metadata` (you can refer to the format of `eval-82-5-1.csv` and `train-82-5-1.csv` and the format of `class_labels_pann-5-1.csv`).

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

### 1.Data processing
Prepare the JSON files for data (i.e. `/ast/egs/audioset/data/datafiles/eval-55-3.json` and `/ast/egs/audioset/data/datafiles/train-55-3.json`) and put them in `/ast/egs/audioset/data/datafiles`. It is also necessary to generate a CSV file for categories, such as `/ast/egs/audioset/data/class_labels_ast-3.csv`.

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

![result-acc](/result-acc.png)

![result-map](/result-mAP.png)

1,406 clear sound data were selected for the experiment, and the following experimental results were obtained:

![result](/result.png)

