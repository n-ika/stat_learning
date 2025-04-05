import torch; torch.manual_seed(108)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import pandas as pd
import seaborn as sns
import librosa
import os
from ae_zevin import *
import sys


n_fft=512
n_mels=16
tgt_sr = 16000

for exp in [1,2]:
    if exp==1:
        root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_one/'
        root_stims = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/zevin_data/stimuli/simulation_one/'
        test_words = ['pa-bi-ku', 'ti-bu-do', 'tu-da-ro', 'pi-go-la']
    elif exp==2:
        root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_two/'
        root_stims = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/zevin_data/stimuli/simulation_two/'
        test_words = ['tu-pi-du', 'go-ge-bi', 'gu-lo-ba', 'bo-ga-ki']

    file_path = root+"training_final.wav"

    def split_audio(file_path, chunk_duration=0.222, start_time=0):
        y, sr = librosa.load(file_path, sr=None, offset=start_time)
        chunk_samples = int(chunk_duration * sr)
        chunks = [y[i:i+chunk_samples] for i in range(0, len(y), chunk_samples)]
        return chunks, sr
    start_time = 0.000 

    chunks, sample_rate = split_audio(file_path, start_time=start_time)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration of each chunk: {chunks[0].shape[0] / sample_rate:.2f} seconds")

    mels = []
    for chunk in chunks:
        mel_specgram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4, n_mels=n_mels, window='hann') 
        mels.append(mel_specgram)

    tr_data = np.array(mels[:-1])
    tr_data_ = np.reshape(tr_data, (tr_data.shape[0],tr_data.shape[1]*tr_data.shape[2]))
    train_data = torch.Tensor(tr_data_)
    sequential_syllables = np.vstack((tr_data_[1:,:],np.zeros(tr_data.shape[1]*tr_data.shape[2])))
    sequential_syllables=torch.Tensor(sequential_syllables)

    # np.save(root+'acoustic_train_data.npy',tr_data_)
    syllables = [k.split('-') for k in test_words]
    syll_vec = {}
    for word in syllables:
        for syll in word:
            y, sr = librosa.load(root_stims+syll+'.wav', sr=None)
            audio = librosa.resample(y, orig_sr=sr, target_sr=tgt_sr)
            mel_specgram = librosa.feature.melspectrogram(y=audio, sr=tgt_sr, n_fft=n_fft, hop_length=n_fft//4, n_mels=n_mels, window='hann')
            # stretched_audio = stretch_audio(audio, tgt_sr, target_length)
            syll_vec[syll]=mel_specgram
    
    np.savez(root+'syll_vec_acoustic', **syll_vec)
    torch.save(train_data, root+'acoustic_train_data.pt')
    torch.save(sequential_syllables, root+'acoustic_train_data_sequential.pt')

