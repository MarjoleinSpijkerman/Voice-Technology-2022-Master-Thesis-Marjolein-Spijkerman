#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=training-on-gpu
#SBATCH --mem=8GB

source /data/s3219305/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate MaskCycleGAN-VC

tensorboard --logdir results/logs &
python -u -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_uaspeech_M05_CM01 \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_training/ \
    --speaker_A_id M05 \
    --speaker_B_id CM01 \
    --epochs_per_save 5 \
    --epochs_per_plot 5 \
    --num_epochs 25 \
    --batch_size 1 \
    --generator_lr 5e-4 \
    --discriminator_lr 5e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 