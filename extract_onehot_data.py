import torch; torch.manual_seed(108)
import numpy as np
import pandas as pd
from ae_zevin import *


def get_train_data(transcript_df, syll_vec):
        data = []
        stats = []
        for _, row in transcript_df.iterrows():
            stim = row['stimulus'].split('-')
            data.append(syll_vec[stim[0]])
            data.append(syll_vec[stim[1]])
            data.append(syll_vec[stim[2]])
            stats.append(stim[0])
            stats.append(stim[1])
            stats.append(stim[2])
            # break
        train_data = np.vstack(data)
        sequential_syllables = np.vstack((train_data[1:,:],np.zeros(12)))

        df=pd.DataFrame([(x,stats[i+1]) for i,x in enumerate(stats[:-1])],columns=['syll','next_syll'])
        df = df.groupby(df.columns.tolist(),as_index=False).size()
        df['emp_tp'] = np.NaN
        for syll in syll_vec.keys():
            # divide the number of transitions from syll by the total number of transitions from syll
            df.loc[df['syll']==syll,'emp_tp'] = df[(df['syll']==syll)]['size'] / df[df['syll']==syll]['size'].sum()
        return(train_data, sequential_syllables, stats, df)

for exp in [1,2]:
    if exp==1:
        root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_one/'
        onehot = F.one_hot(torch.arange(0, 12).view(12,1), num_classes=12)
        ex1 = {'pa-bi-ku', 'ti-bu-do', 'go-la-tu', 'da-ro-pi'}
        ex2 = {'gu-lo-ba', 'bo-ga-ki', 'ge-bi-tu', 'pi-du-go'}
        pa = onehot[0]
        bi = onehot[1]
        ku = onehot[2]
        ti = onehot[3]
        bu = onehot[4]
        do = onehot[5]
        go = onehot[6]
        la = onehot[7]
        tu = onehot[8]
        da = onehot[9]
        ro = onehot[10]
        pi = onehot[11]
        syll_vec = {'pa':pa,'bi':bi,'ku':ku,'ti':ti,'bu':bu,'do':do,'go':go,'la':la,'tu':tu,'da':da,'ro':ro,'pi':pi}

    elif exp==2:
        root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_two/'
        onehot = F.one_hot(torch.arange(0, 12).view(12,1), num_classes=12)
        gu = onehot[0]
        lo = onehot[1]
        ba = onehot[2]
        bo = onehot[3]
        ga = onehot[4]
        ki = onehot[5]
        ge = onehot[6]
        bi = onehot[7]
        tu = onehot[8]
        pi = onehot[9]
        du = onehot[10]
        go = onehot[11]
        syll_vec = {'gu':gu,'lo':lo,'ba':ba,'bo':bo,'ga':ga,'ki':ki,'ge':ge,'bi':bi,'tu':tu,'pi':pi,'du':du,'go':go}

    transcript_df = pd.read_csv(root+'transcript_train.csv',header=None)
    transcript_df.columns = ['stimulus']

    tr_data, sequential, stats, tp_df = get_train_data(transcript_df, syll_vec)

    train_data = torch.Tensor(tr_data)
    sequential_syllables=torch.Tensor(sequential)
    # np.save(root+'onehot_train_data.npy',tr_data)
    tp_df.to_csv(root+'tp_df.csv')
    np.save(root+'stats.npy',stats)
    np.savez(root+'syll_vec_onehot', **syll_vec)
    torch.save(train_data, root+'onehot_train_data.pt')
    torch.save(sequential_syllables, root+'onehot_train_data_sequential.pt')

