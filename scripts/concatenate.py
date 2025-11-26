import os
import pandas as pd
import sys

model_type=sys.argv[1]
out_name = sys.argv[2]

root_in = f'/projects/jurovlab/stat_learning/{model_type}_results_{out_name}/'
root_out = f'/projects/jurovlab/stat_learning/results/'
os.makedirs(root_out, exist_ok=True)

def concatenate_dfs(root):
    all_dfs = []
    for file in os.listdir(root):
        if file.endswith('.csv') and not file.startswith('loss'):
            if os.path.getsize(os.path.join(root, file)) > 0:
                try:
                    df = pd.read_csv(os.path.join(root, file), compression='gzip')
                except Exception as e:
                    df = pd.read_csv(os.path.join(root, file))
                df['condition'] = df['condition'].map({1: 'Word', 0: 'Part-Word'})
                all_dfs.append(df)
    return pd.concat(all_dfs)


df_bigram = concatenate_dfs(root_in+'bigram_data/out/')
df_unigram = concatenate_dfs(root_in+'unigram_data/out/')
df_zerobigram = concatenate_dfs(root_in+'zerovec-bigram_data/out/')

df_unigram['btc_ep']=df_unigram['batch_id']+(df_unigram['epochs']-1)*809
df_bigram['btc_ep']=df_bigram['batch_id']+(df_bigram['epochs']-1)*809
df_zerobigram['btc_ep']=df_zerobigram['batch_id']+(df_zerobigram['epochs']-1)*809

df_unigram.to_csv(root_out+f'/{model_type}_unigram_{out_name}.csv',compression='gzip')
df_zerobigram.to_csv(root_out+f'/{model_type}_zerobigram_{out_name}.csv',compression='gzip')
df_bigram.to_csv(root_out+f'/{model_type}_bigram_{out_name}.csv',compression='gzip')

# df_all = pd.concat([df_bigram, df_unigram, df_zerobigram])
# df_all.to_csv(root_out+f'/final_{out_name}.csv',compression='gzip')
