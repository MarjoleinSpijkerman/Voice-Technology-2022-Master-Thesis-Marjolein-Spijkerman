#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=testing-maskcycleGAN-VC-M05-GPU
#SBATCH --mem=8GB

source /data/s3219305/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate MaskCycleGAN-VC


python -u -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name 'experiments_M5/mask_cyclegan_uaspeech_M05_0.75x_CM01' \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_experiments_M5 \
    --gpu_ids 0 \
    --speaker_A_id 'M05_0.75x' \
    --speaker_B_id CM01 \
    --ckpt_dir /data/s3219305/MaskCycleGAN-VC/results/mask_cyclegan_uaspeech_M05_CM01/ckpts/ \
    --load_epoch 25 \
    --model_name generator_A2B

python -u -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name 'experiments_M5/mask_cyclegan_uaspeech_M05_1.0x_CM01' \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_experiments_M5 \
    --gpu_ids 0 \
    --speaker_A_id 'M05_1.0x' \
    --speaker_B_id CM01 \
    --ckpt_dir /data/s3219305/MaskCycleGAN-VC/results/mask_cyclegan_uaspeech_M05_CM01/ckpts/ \
    --load_epoch 25 \
    --model_name generator_A2B

python -u -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name 'experiments_M5/mask_cyclegan_uaspeech_M05_1.25x_CM01' \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_experiments_M5 \
    --gpu_ids 0 \
    --speaker_A_id 'M05_1.25x' \
    --speaker_B_id CM01 \
    --ckpt_dir /data/s3219305/MaskCycleGAN-VC/results/mask_cyclegan_uaspeech_M05_CM01/ckpts/ \
    --load_epoch 25 \
    --model_name generator_A2B

python -u -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name 'experiments_M5/mask_cyclegan_uaspeech_M05_1.5x_CM01' \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_experiments_M5 \
    --gpu_ids 0 \
    --speaker_A_id 'M05_1.5x' \
    --speaker_B_id CM01 \
    --ckpt_dir /data/s3219305/MaskCycleGAN-VC/results/mask_cyclegan_uaspeech_M05_CM01/ckpts/ \
    --load_epoch 25 \
    --model_name generator_A2B

python -u -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name 'experiments_M5/mask_cyclegan_uaspeech_M05_1.75x_CM01' \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_experiments_M5 \
    --gpu_ids 0 \
    --speaker_A_id 'M05_1.75x' \
    --speaker_B_id CM01 \
    --ckpt_dir /data/s3219305/MaskCycleGAN-VC/results/mask_cyclegan_uaspeech_M05_CM01/ckpts/ \
    --load_epoch 25 \
    --model_name generator_A2B
    
python -u -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name 'experiments_M5/mask_cyclegan_uaspeech_M05_2.0x_CM01' \
    --save_dir results/ \
    --preprocessed_data_dir UASpeech_preprocessed/uaspeech_experiments_M5 \
    --gpu_ids 0 \
    --speaker_A_id 'M05_2.0x' \
    --speaker_B_id CM01 \
    --ckpt_dir /data/s3219305/MaskCycleGAN-VC/results/mask_cyclegan_uaspeech_M05_CM01/ckpts/ \
    --load_epoch 25 \
    --model_name generator_A2B