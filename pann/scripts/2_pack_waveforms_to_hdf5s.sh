#!/bin/bash
DATASET_DIR=${1:-"/home/cn/lcr/pann/audioset_tagging_cnn-master"}   # Default first argument.
WORKSPACE=${2:-"/home/cn/lcr/pann/audioset_tagging_cnn-master/workspaces/pann-82-5-20"}   # Default second argument.

# Pack evaluation waveforms to a single hdf5 file
python3 /home/cn/lcr/pann/audioset_tagging_cnn-master/utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$DATASET_DIR"/metadata/eval-82-5-1.csv" \
    --audios_dir=$DATASET_DIR"/audioset/eval-82-5" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5"

# Pack balanced training waveforms to a single hdf5 file
python3 /home/cn/lcr/pann/audioset_tagging_cnn-master/utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$DATASET_DIR"/metadata/train-82-5-1.csv" \
    --audios_dir=$DATASET_DIR"/audioset/train-82-5" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/train.h5"
:<<!
# Pack unbalanced training waveforms to hdf5 files. Users may consider 
# executing the following commands in parallel to speed up. One simple 
# way is to open 41 terminals and execute one command in one terminal.
for IDX in {00..40}; do
    echo $IDX
    python3 utils/dataset.py pack_waveforms_to_hdf5 \
        --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" \
        --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" \
        --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5"
done
!
