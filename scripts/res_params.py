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
import warnings
warnings.filterwarnings('ignore')


hidden_fixed=True

if hidden_fixed==True:
    type_exp = 'changing_bottleneck_width/'
    bottleneck_sizes=[8,16,32,64,128,256]
    num_layers=[7]
else:
    type_exp = 'changing_layer_numbers/'
    bottleneck_sizes=[8]
    num_layers=[1,2,3,4,5,6,7]

epoch = 1
NUMs = np.arange(5)
batch_sizes = [1,2,4,8,16,32,64,128]
LRs = [0.0001, 0.01, 1e-05, 0.001]
exps = [1,2]


model_types = ['onehot','acoustic','phon','acoustic_norm','acoustic_new','acoustic_new_norm']
# model_types = ['onehot','acoustic','phon']
root_test = f'/projects/jurovlab/stat_learning/models_testing_params/{type_exp}'
os.makedirs(root_test+"figs",exist_ok=True)

dfs = []
for model_type in model_types:
    for num_layer in num_layers:
        for exp in exps:
            for bottleneck_size in bottleneck_sizes:
                for LR in LRs:
                    for NUM in NUMs:
                        for batch_size in batch_sizes:
                            eval_path = root_test+f'{model_type}_{bottleneck_size}_{num_layer}/eval_ep-{epoch}_exp-{exp}_n-{NUM}_lr-{LR}_btc-{batch_size}.csv'
                            try:
                                df = pd.read_csv(eval_path,header=0)
                                df['model_type']=f'{model_type}_{bottleneck_size}_{num_layer}'
                                df['exp']=exp
                                df['epoch']=epoch
                                df['NUM']=NUM
                                df['LR']=LR
                                df['batch_size']=batch_size
                                df['bottleneck_size']=bottleneck_size
                                df['num_layer']=num_layer
                                dfs.append(df)
                            except FileNotFoundError:
                                print(eval_path)
                                pass
                            

eval_df = pd.concat(dfs)
del eval_df['Unnamed: 0']
eval_df.to_csv(root_test+'param_results.csv')
