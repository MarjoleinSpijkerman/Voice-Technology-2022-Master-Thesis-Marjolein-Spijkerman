# -*- coding: utf-8 -*-
"""
Preprocesses .wav to Mel-spectrograms using Mel-GAN vocoder and saves them to pickle files.
MelGAN vocoder: https://github.com/descriptinc/melgan-neurips
"""

import os
import argparse
import pickle
import glob
import random
import numpy as np
from tqdm import tqdm

import librosa
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

SAMPLING_RATE = 22050  # Fixed sampling rate


def normalize_mel(wavspath, speedrate):
    wav_files = glob.glob(os.path.join(
        wavspath, '**', '*.wav'), recursive=True)  # source_path
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    
    mel_list = list()
    wav_list = list()
    for wavpath in tqdm(wav_files, desc='Preprocess wav to mel'):
        wav_orig, _ = librosa.load(wavpath, sr=SAMPLING_RATE, mono=True)
        
        '''This part is used for the time-stretching. See https://librosa.org/doc/main/generated/librosa.phase_vocoder.html'''
        D = librosa.stft(wav_orig, hop_length=512)
        D_fast  = librosa.phase_vocoder(D, rate=speedrate, hop_length=512)
        y_fast  = librosa.istft(D_fast, hop_length=512)
        wav_orig = y_fast
		'''End of time-stretching'''
		
        spec = vocoder(torch.tensor([wav_orig]))
        
        wav_list.append(wavpath)
        if spec.shape[-1] >= 64:    # training sample consists of 64 randomly cropped frames
            mel_list.append(spec.cpu().detach().numpy()[0])

    mel_concatenated = np.concatenate(mel_list, axis=1)
    mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
    mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

    mel_normalized = list()
    for mel in mel_list:
        assert mel.shape[-1] >= 64, f"Mel spectogram length must be greater than 64 frames, but was {mel.shape[-1]}"
        app = (mel - mel_mean) / mel_std
        mel_normalized.append(app)

    return mel_normalized, mel_mean, mel_std, wav_list


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_dataset(data_path, speaker_id, speedrate, cache_folder='./cache/'):
    """Preprocesses dataset of .wav files by converting to Mel-spectrograms.

    Args:
        data_path (str): Directory containing .wav files of the speaker.
        speaker_id (str): ID of the speaker.
        cache_folder (str, optional): Directory to hold preprocessed data. Defaults to './cache/'.
    """

    print(f"Preprocessing data for speaker: {speaker_id}.")

    mel_normalized, mel_mean, mel_std, wav_list = normalize_mel(data_path, speedrate)

    if not os.path.exists(os.path.join(cache_folder, f"{speaker_id}_{speedrate}x")):
        os.makedirs(os.path.join(cache_folder, f"{speaker_id}_{speedrate}x"))

    np.savez(os.path.join(cache_folder, f"{speaker_id}_{speedrate}x", f"{speaker_id}_norm_stat.npz"),
             mean=mel_mean,
             std=mel_std)

    save_pickle(variable=mel_normalized,
                fileName=os.path.join(cache_folder, f"{speaker_id}_{speedrate}x", f"{speaker_id}_normalized.pickle"))
    
	'''Saving the order of the used audio files. This is used for later creating a list of the original transcriptions'''
    save_loc = os.path.join(cache_folder, f"{speaker_id}_{speedrate}x", f"{speaker_id}_wav_list.txt")
    with open(save_loc, "w") as output:
        for item in wav_list:
            output.write(str(item))
            output.write("\n")
	'''end of added changes'''

    print(f"Preprocessed and saved data for speaker: {speaker_id}_{speedrate}x.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_directory', type=str, default='vcc2018/vcc2018_training',
                        help='Directory holding VCC2018 dataset.')
    parser.add_argument('--preprocessed_data_directory', type=str, default='vcc2018_preprocessed/vcc2018_training',
                        help='Directory holding preprocessed VCC2018 dataset.')
    parser.add_argument('--speaker_ids', nargs='+', type=str, default=['VCC2SM3', 'VCC2TF1'],
                        help='Source speaker id from VCC2018.')
	'''here an additional argument is added for the time-stretching speed rate'''
    parser.add_argument('--rate', type=float, default = 1, help='time stretching')

    args = parser.parse_args()

    for speaker_id in args.speaker_ids:
        data_path = os.path.join(args.data_directory, speaker_id)
        preprocess_dataset(data_path=data_path, speaker_id=speaker_id, speedrate=args.rate,
                           cache_folder=args.preprocessed_data_directory)
