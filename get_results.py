import torch; torch.manual_seed(108)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
import pandas as pd
import seaborn as sns
import librosa
import os
from ae_zevin import *
import sys

model_type = sys.argv[1]

root_path = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/'
os.makedirs(root_path+'results',exist_ok=True)

hidden_size = 512
num_layers = 7


def extract_errors(syll1,syll2,model=None):
    # take two syllables (as torch tensors) and compute MSE between them
    mse = torch.nn.MSELoss()
    syll1 = syll1.to(torch.float32)
    encoded,recon = model(syll1)
    mse_r = mse(recon,syll2).detach().numpy()
    recon = recon.detach().numpy()
    return(recon,mse_r)

dfs = []
for exp in [1,2]:
    if exp==1:
        root = root_path+'simulation_one/'
        test_words = ['pa-bi-ku', 'ti-bu-do', 'tu-da-ro', 'pi-go-la']
    elif exp==2:
        root = root_path+'simulation_two/'
        test_words = ['tu-pi-du', 'go-ge-bi', 'gu-lo-ba', 'bo-ga-ki']

    tp_df = pd.read_csv(root+'tp_df.csv')
    root_models = root +'models/'+model_type+'/'
    train_data = torch.load(root+f'{model_type}_train_data.pt',weights_only=True)


    if model_type == 'onehot':
        # hidden_size = 256
        # num_layers = 5
        syll_vec_ = np.load(root+'syll_vec_onehot.npz', allow_pickle=True)
        syll_vec = {}
        for key in syll_vec_.keys():
            syll_vec[key] = torch.tensor(syll_vec_[key])
    elif model_type == 'phon':
        # hidden_size = 256
        # num_layers = 5
        syll_vec_ = np.load(root+'syll_vec_phon.npz', allow_pickle=True)
        syll_vec = {}
        for key in syll_vec_.keys():
            syll_vec[key] = torch.tensor(np.reshape(syll_vec_[key].astype(np.float32), (syll_vec_[key].shape[0]*syll_vec_[key].shape[1])))
    elif model_type == 'acoustic':
        syll_vec_ = np.load(root+'syll_vec_acoustic.npz', allow_pickle=True)
        # hidden_size = 512
        # num_layers = 6
        syll_vec = {}
        for key in syll_vec_.keys():
            syll_vec[key] = torch.tensor(np.reshape(syll_vec_[key], (syll_vec_[key].shape[0]*syll_vec_[key].shape[1])))

    for epoch_num in np.logspace(0, 13, num=14, base=2, dtype=int):
        for m_num in range(50):
            res = {'tp':[],'condition':[],'model_num':[],'ae_mse':[],'word':[]}
            model = AE(input_shape=train_data.shape[1], hidden_size=hidden_size, num_layers=num_layers)
            model.load_state_dict(torch.load(root_models+f'ckpt_{epoch_num}_exp-{exp}_{m_num}.pt',weights_only=True))
            model.eval() 
            for word in test_words:
                sylls =  word.split('-')
                recon, mse_r1 = extract_errors(syll_vec[sylls[0]], syll_vec[sylls[1]], model=model)
                recon, mse_r2 = extract_errors(syll_vec[sylls[1]], syll_vec[sylls[2]], model=model)
                res['model_num'].append(m_num)
                res['word'].append(word)
                if tp_df[(tp_df.syll==sylls[0])&(tp_df.next_syll==sylls[1])].emp_tp.empty:
                    tp1 = 0
                else:
                    tp1 = float(tp_df[(tp_df.syll==sylls[0])&(tp_df.next_syll==sylls[1])].emp_tp.iloc[0])
                if tp_df[(tp_df.syll==sylls[1])&(tp_df.next_syll==sylls[2])].emp_tp.empty:
                    tp2 = 0
                else:
                    tp2 =float(tp_df[(tp_df.syll==sylls[1])&(tp_df.next_syll==sylls[2])].emp_tp.iloc[0])
                tp_avg = np.average((tp1,tp2))
                res['tp'].append(tp_avg)
                res['ae_mse'].append(np.average((mse_r1,mse_r2)))
                if exp==1:
                    if word == 'pa-bi-ku' or word=='ti-bu-do':
                        res['condition'].append('word')
                    else:
                        res['condition'].append('part-word')
                elif exp==2:
                    if word == 'gu-lo-ba' or word=='bo-ga-ki':
                        res['condition'].append('word')
                    else:
                        res['condition'].append('part-word')
            df=pd.DataFrame(res)
            df['epochs']=epoch_num
            df['exp']=exp
            df['model_type']=model_type
            dfs.append(df)

df_all = pd.concat(dfs)
df_all.to_csv(root_path+f'results/res_{model_type}.csv')

# test_words = ['pa-bi-ku', 'ti-bu-do', 'tu-da-ro', 'pi-go-la']
# test_words = ['gu-lo-ba', 'bo-ga-ki', 'go-ge-bi', 'tu-pi-du']
