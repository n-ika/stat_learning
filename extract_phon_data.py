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


encoding_zevin = pd.read_csv('/Users/nika/Desktop/hippocampus/stat_learning/zevin/zevin_data/language_simulations/mapping.txt',
            header=None,sep='\s+')

for exp in [1,2]:
    if exp==1:
        root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_one/'
    elif exp==2:
        root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_two/'
    stats = np.load(root+'stats.npy')
    # make encodings into a dictionary
    encodings = {}
    for i in range(encoding_zevin.shape[0]):
        encodings[encoding_zevin.loc[i,0]] = encoding_zevin.loc[i,1:].values
    syll_vec = {}
    encoding_train = []
    for itm in stats:
        encoding_train.append(np.vstack([encodings[itm[0]],encodings[itm[1]]]))
        syll_vec[itm] = np.vstack([encodings[itm[0]],encodings[itm[1]]])

    test_items =['pa','bi','ku','ti','bu','do','tu','da','ro','pi','go','la','tu','pi','du','go','ge','bi','gu','lo','ba','bo','ga','ki']
    for itm in test_items:
        if itm not in syll_vec.keys():
            syll_vec[itm] = np.vstack([encodings[itm[0]],encodings[itm[1]]])

    encoding_train = np.stack(encoding_train,axis=0)
    tr_data = encoding_train.reshape(-1,2*25)
    sequential  = np.vstack((tr_data[1:,:],np.zeros(50)))

    train_data = torch.Tensor(tr_data.astype(np.float32))
    sequential_syllables=torch.Tensor(sequential.astype(float))
    torch.save(train_data, root+'phon_train_data.pt')
    torch.save(sequential_syllables, root+'phon_train_data_sequential.pt')
    # np.save(root+'phon_train_data.npy',tr_data.astype(np.float32))
    np.savez(root+'syll_vec_phon', **syll_vec)