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
N=1

if hidden_fixed==True:
    type_exp = f'changing_bottleneck_width_{N}/'
else:
    type_exp = f'changing_layer_numbers_{N}/'

root_test = f'/projects/jurovlab/stat_learning/models_testing_params/{type_exp}'
eval_df=pd.read_csv(root_test+'param_results.csv')

for name_model in eval_df.model_type.unique():
    modeltype = ' '.join(name_model.split('_')[:-1])
    bttlneck = name_model.split('_')[-2]
    num_layers = name_model.split('_')[-1]
    plt.figure(figsize=(6, 4))
    g = sns.relplot(data=eval_df[eval_df.model_type==name_model],
                x='LR', 
                y='loss',
                hue='batch_size',
                row='exp',
                style='batch_size',
                kind='line',
                # sizes=(20, 40),
                facet_kws={'sharey': False},
                palette='Set2',
                # dodge=True,
                )
    # sns.violinplot(data=df_filt, x='condition', y='totallook',hue='condition',hue_order=['word', 'partword'])
    g.set_titles("Experiment {row_name}")  # Set titles for each subplot
    plt.suptitle(f'Model: {modeltype}, layers #: {num_layers}, bottleneck #: {bttlneck}, 5 models per params', y=1.02)  # Add a global title
    for ax in g.axes.flatten():

        ax.set_xscale('log')
    plt.savefig(root_test+f"figs/{name_model}.png", dpi=300, bbox_inches='tight')
    plt.close()