import numpy as np
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

ARG1 = sys.argv[1]
CORR = sys.argv[2]


print(f'SIZE NET: {ARG1}')
print(f'CORR: {CORR}')

if CORR == 'True':
    human_data = pd.read_csv('/projects/jurovlab/stat_learning/data/human_corrected.csv')
    addon = '_corrected'
    print('Using corrected human data')
else:
    human_data = pd.read_csv('/projects/jurovlab/stat_learning/data/human_data.csv')
    addon = ''

human_data['condition'] = human_data['condition'].map({'word': 'Word', 'part_word': 'Part-Word'})
root = f'/projects/jurovlab/stat_learning/results/results_{ARG1}/rnn_unigram_mse.csv'
N=1



def get_item_familiarity_rankings(x, items):
    # order is given relative to order of items list
    # default is increasing order of unfamiliarity, so the first item in the order is the most familiar
    res = []
    for item in items:
        # print(x[x['stimulus']==item])
        assert len(x[x['stimulus']==item]) == 1
        res.append(x[x['stimulus']==item]['response'].values[0])
    order = np.argsort(res,kind='stable')
    equalities=[]
    for i in range(1,len(order)):
        if res[order[i]]==res[order[i-1]]:
            equalities.append(i)
    return tuple(order), tuple(equalities)


def get_item_familiarity_rankings_by_subject(data, items):
    f = lambda x: get_item_familiarity_rankings(x, items)
    return data.groupby(['subject'])[['stimulus','response']].apply(f)


def get_average_item_ranks(rankings):
    nb_items = len(rankings.iloc[0][0])
    average_item_ranks = []
    for item in range(nb_items):
        item_ranks = []
        for order, equalities in rankings:
            item_rank = order.index(item)
            equal_ranks = [item_rank]
            for j in range(item_rank, 0,-1):
                if j in equalities:
                    equal_ranks.append(j-1)
                else:
                    break
            for j in range(item_rank+1, nb_items):
                if j in equalities:
                    equal_ranks.append(j)
                else:
                    break
            item_rank = np.mean(np.array(equal_ranks)+1)  # adding one for having rank starting at one (most familiar)
            item_ranks.append(item_rank)
        average_item_ranks.append(np.mean(item_ranks))
    return average_item_ranks

def prepare_xp_results(xp_human_data, xp_model_data, xp_items):
    human_rankings = get_item_familiarity_rankings_by_subject(xp_human_data, xp_items)
    human_avg_ranks = get_average_item_ranks(human_rankings)
    human_results = pd.DataFrame({'item': xp_items, 'average_rank': human_avg_ranks})
    model_results = {'item': [], 'average_rank': [], 'model_type': [], 'nb_training_btc_ep': []}
    for (model_type, nb_epochs), group_df in xp_model_data.groupby(['model_type', 'btc_ep']):
        rankings = get_item_familiarity_rankings_by_subject(group_df, xp_items)
        avg_ranks = get_average_item_ranks(rankings)
        model_results['average_rank'] = model_results['average_rank'] + avg_ranks
        model_results['item'] = model_results['item'] + xp_items
        model_results['model_type'] = model_results['model_type'] + [model_type]*len(xp_items)
        model_results['nb_training_btc_ep'] = model_results['nb_training_btc_ep'] + [nb_epochs]*len(xp_items)
    
    model_results = pd.DataFrame(model_results)
    return human_results, model_results



df_orig = pd.read_csv(root, compression='gzip', index_col=0)
df_orig['btc_ep'] = [0 if epoch==0 else (epoch-1)*809*N+batch+1 for batch,epoch in zip(df_orig['batch_id'], df_orig['epochs'])]

del df_orig['lr']
del df_orig['batch_id']
# del df_orig['epochs']
del df_orig['stim_type']

df_orig1 = df_orig[df_orig.exp==1].copy()
df_orig2 = df_orig[df_orig.exp==2].copy()

df_orig1['stimulus'] = df_orig1['in_label'].map({'pa': 'pabiku', 
                                               'bi': 'pabiku', 
                                               'ti': 'tibudo', 
                                               'bu': 'tibudo',
                                               'tu': 'tudaro',
                                               'da': 'tudaro',
                                               'pi': 'pigola',
                                               'go': 'pigola',
                                               })
df_orig2['stimulus'] = df_orig2['in_label'].map({'gu': 'guloba', 
                                               'lo': 'guloba', 
                                               'bo': 'bogaki', 
                                               'ga': 'bogaki',
                                               'tu': 'tupidu',
                                               'pi': 'tupidu',
                                               'ge': 'gogebi',
                                               'go': 'gogebi',
                                               })


df_orig1['response'] = df_orig1['loss']  # lower is more familiar, as for humans
df_orig2['response'] = df_orig2['loss']  # lower is more familiar, as for humans
df_orig1['subject'] = df_orig1['model_num']
df_orig2['subject'] = df_orig2['model_num']

xp1_model_data = df_orig1.groupby(['btc_ep', 'model_type', 'subject',  'stimulus', ],
                 as_index=False)[['response']].mean()

xp2_model_data = df_orig2.groupby(['btc_ep', 'model_type', 'subject',  'stimulus',],
                 as_index=False)[['response']].mean()


xp1_items = ['pabiku', 'tibudo', 'pigola', 'tudaro']  # first two are words
xp1_human_data = human_data[human_data['experiment']==1]

xp2_items = ['bogaki', 'guloba', 'tupidu', 'gogebi']  # first two are words
xp2_human_data = human_data[human_data['experiment']==2]

hum_xp1, mod_xp1 = prepare_xp_results(xp1_human_data, xp1_model_data, xp1_items)
hum_xp2, mod_xp2 = prepare_xp_results(xp2_human_data, xp2_model_data, xp2_items)

hum_xp1.to_csv(f'/projects/jurovlab/stat_learning/results/rank_analysis/hum_xp1_{ARG1}{addon}.csv', index=False)
mod_xp1.to_csv(f'/projects/jurovlab/stat_learning/results/rank_analysis/mod_xp1_{ARG1}{addon}.csv', index=False)
hum_xp2.to_csv(f'/projects/jurovlab/stat_learning/results/rank_analysis/hum_xp2_{ARG1}{addon}.csv', index=False)
mod_xp2.to_csv(f'/projects/jurovlab/stat_learning/results/rank_analysis/mod_xp2_{ARG1}{addon}.csv', index=False)