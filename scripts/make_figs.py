from logging import root
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

model_type=sys.argv[1]
loss_type=sys.argv[2]  # 'mse' or 'ce'

root_out = f'/projects/jurovlab/stat_learning/results/stats/'
root_in=f'/projects/jurovlab/stat_learning/results/'
os.makedirs(root_out, exist_ok=True)
os.makedirs(root_out+'plots/', exist_ok=True)

def plot_lm(df, stim_type, EXP, root_out=root_out, loss_type=loss_type):
    g = sns.lmplot(data=df[df.exp==EXP],
            x='btc_ep',
            y='loss',
            lowess=True,
            col='model_type',
            hue='condition',
            hue_order=['Word','Part-Word'],
            # row='exp',
            col_order=['onehot','phon','acoustic_new_norm'],
            facet_kws={"sharey": False}
            )
    plt.suptitle(f'Experiment {EXP} -  Results - stimulus type: {stim_type} - 70 models',size=20,y=1.05,x=0.55)
    g.set_titles(col_template="{col_name}")
    g.set_xlabels('Batch x Epoch')
    g.set_ylabels(f'{loss_type.upper()} Loss')
    titles = ['Onehot', 'Phoneme Embedding', 'Acoustic Features']
    for ax, title in zip(g.axes[0], titles):  # for columns
        ax.set_title(title)
    plt.savefig(root_out+f'plots/{stim_type}_exp{EXP}_lmplot.png',bbox_inches='tight', 
                dpi=300)
    plt.close()


def plot_line(df, stim_type, root_out=root_out, loss_type=loss_type):
    g = sns.relplot(data=df,
            x='btc_ep',
            y='loss',
            hue='condition',
            row='exp',
            col='model_type',
            col_order=['onehot','phon','acoustic_new_norm'],
            hue_order=['Word','Part-Word'],
            style='stat_type',
            style_order=['median', 'min', 'max', 'q1', 'q3'],
            kind='line',
            facet_kws={"sharey": False}
            )
    plt.suptitle(f'Experiment 1&2 - Results - stimulus type: {stim_type} - 70 models',size=20,y=1.05,x=0.55)
    g.set_titles(col_template="{col_name}")
    g.set_xlabels('Batch x Epoch')
    g.set_ylabels(f'{loss_type.upper()} Loss')
    titles = ['Onehot', 'Phoneme Embedding', 'Acoustic Features']
    for ax, title in zip(g.axes[0], titles):  # for columns
        ax.set_title(title)
    plt.savefig(root_out+f'plots/{stim_type}_btcep.png',bbox_inches='tight', 
                dpi=300)
    plt.close()

id_cols = ['model_type','exp',
        'condition','epochs',
        'in_label','out_label',
        'batch_id','btc_ep',
        'stim_type']

def get_stat_point(df,cols=id_cols):
    df_stat = df.groupby(cols).agg(median=("loss", "median"),
                                            min=("loss", "min"),
                                            max=("loss", "max"),
                                            q1=("loss", lambda x: x.quantile(0.25)),
                                            q3=("loss", lambda x: x.quantile(0.75))
                                            ).reset_index()
    return df_stat


for stim_type in ['unigram','bigram','zerobigram']:
    df = pd.read_csv(root_in+f'{model_type}_{stim_type}_{loss_type}.csv',compression='gzip')
    df['btc_ep']=df['batch_id']+(df['epochs'])*809
    df.loc[df['epochs']>=1,'btc_ep'] -= 808
    df_stat = get_stat_point(df)
    df_melt = pd.melt(df_stat, 
        id_vars=id_cols,
        value_vars=["median", "min", "max", "q1", "q3"], 
        var_name="stat_type", 
        value_name="loss")
    df_melt.to_csv(root_out+f'{stim_type}_{model_type}_df.csv',compression='gzip')
    # plot_line(df_melt, stim_type)

        


