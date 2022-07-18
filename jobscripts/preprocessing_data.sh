#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=preprocessing_the_data
#SBATCH --mem=8GB

source /data/s3219305/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate MaskCycleGAN-VC

python data_preprocessing/preprocess_vcc2018_v2.py \
  --data_directory UASpeech/uaspeech_evaluation \
  --preprocessed_data_directory UASpeech_preprocessed/uaspeech_experiments_M5\
  --speaker_ids M05 \
  --rate 1.0

python data_preprocessing/preprocess_vcc2018_v2.py \
  --data_directory UASpeech/uaspeech_evaluation \
  --preprocessed_data_directory UASpeech_preprocessed/uaspeech_experiments_M5\
  --speaker_ids M05 \
  --rate 0.75

python data_preprocessing/preprocess_vcc2018_v2.py \
  --data_directory UASpeech/uaspeech_evaluation \
  --preprocessed_data_directory UASpeech_preprocessed/uaspeech_experiments_M5\
  --speaker_ids M05 \
  --rate 1.25

python data_preprocessing/preprocess_vcc2018_v2.py \
  --data_directory UASpeech/uaspeech_evaluation \
  --preprocessed_data_directory UASpeech_preprocessed/uaspeech_experiments_M5\
  --speaker_ids M05 \
  --rate 1.5

python data_preprocessing/preprocess_vcc2018_v2.py \
  --data_directory UASpeech/uaspeech_evaluation \
  --preprocessed_data_directory UASpeech_preprocessed/uaspeech_experiments_M5\
  --speaker_ids M05 \
  --rate 1.75
  
python data_preprocessing/preprocess_vcc2018_v2.py \
  --data_directory UASpeech/uaspeech_evaluation \
  --preprocessed_data_directory UASpeech_preprocessed/uaspeech_experiments_M5\
  --speaker_ids M05 \
  --rate 2.0