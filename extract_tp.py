import numpy as np
import pandas as pd
import os
import sys

exp = int(sys.argv[1])

if exp==1:
    root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_one/'
elif exp==2:
    root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_two/'

if exp==1:
    known_words = {'pa-bi-ku', 'ti-bu-do', 'go-la-tu', 'da-ro-pi'}
    tp = {'syll':[],'next_syll':[],'tp':[],'condition':[]}
    for word in known_words:
        for i,syll in enumerate(word.split('-')):
            if i < 2:
                tp['syll'].append(syll)
                tp['tp'].append(1)
                tp['next_syll'].append(word.split('-')[i+1])
                tp['condition'].append('word')
            else:
                for n_syll in ['pa','ti','go','da']:
                    if n_syll != syll:
                        tp['syll'].append(syll)
                        tp['tp'].append(0.333)
                        tp['next_syll'].append(n_syll)
                        tp['condition'].append('part-word')
                for n_syll in ['bi','bu','ro','la']:
                    if n_syll != syll:
                        tp['syll'].append(syll)
                        tp['tp'].append(0)
                        tp['next_syll'].append(n_syll)
                        tp['condition'].append('none')
    tp_df = pd.DataFrame(tp)

elif exp==2:
    known_words = {'gu-lo-ba', 'bo-ga-ki', 'ge-bi-tu', 'pi-du-go'}
    tp = {'syll':[],'next_syll':[],'tp':[],'condition':[]}
    for word in known_words:
        for i,syll in enumerate(word.split('-')):
            if i < 2:
                tp['syll'].append(syll)
                tp['tp'].append(1)
                tp['next_syll'].append(word.split('-')[i+1])
                tp['condition'].append('word')
            else:
                for n_syll in ['gu','bo','ge','pi']:
                    if n_syll != syll:
                        tp['syll'].append(syll)
                        tp['tp'].append(0.333)
                        tp['next_syll'].append(n_syll)
                        tp['condition'].append('part-word')
                for n_syll in ['lo','bi','du','ga']:
                    if n_syll != syll:
                        tp['syll'].append(syll)
                        tp['tp'].append(0)
                        tp['next_syll'].append(n_syll)
                        tp['condition'].append('none')
    tp_df = pd.DataFrame(tp)

tpdf = pd.read_csv(root+'onehot_tp_df.csv')
merged_df = tp_df.merge(tpdf, on=['syll', 'next_syll'], how='outer')
del merged_df['Unnamed: 0']
merged_df.to_csv(root+'tp_df.csv', index=False)