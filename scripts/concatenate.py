import os
import pandas as pd
import argparse
from multiprocessing import Pool
from functools import partial

def mean_per_group(df):
    # Get one mean loss per batch id
    # There are 28 times more data points for acoustic model than phonological model
    # so we average over batch id to reduce the file size
    df = df.groupby(['condition','model_num','model_type','exp','stim_type','epochs','batch_id'],
                    as_index=False).agg({'loss':'mean'}).reset_index()
    return(df)

def process_file(root, file, do_stats):
    path = os.path.join(root, file)
    try:
        df = pd.read_csv(path, compression="gzip")
    except Exception:
        df = pd.read_csv(path)
    df["condition"] = df["condition"].map({1: "Word", 0: "Part-Word"})
    df["btc_ep"] = df["batch_id"] + (df["epochs"] - 1) * 809
    if do_stats==True:
        return mean_per_group(df)
    else:
        return df

def concatenate_dfs(root, filter_type, NUM_PROCS, do_stats):
    files = [
        f for f in os.listdir(root)
        if f.endswith(".csv")
        and not f.startswith("loss")
        and (filter_type is None or filter_type in f)
        # and (filter_type not in f) #FIXME TODO if want everything BUT filter

    ]
    with Pool(processes=NUM_PROCS) as pool:
        dfs = pool.map(partial(process_file, root, do_stats=do_stats), files)
    return pd.concat(dfs, ignore_index=True)

def process_dfs(args):
    model_type = args.model_type
    out_name = args.out_name
    filter_type = args.filter_type
    encoding_types = args.encoding_type.split(',')
    root_out = args.out_root
    root_in = args.in_root
    do_stats = args.do_stats
    if filter_type is None:
        additional = ''
    else:
        additional = f'_{filter_type}' #TODO
        # additional = ''

    os.makedirs(root_out, exist_ok=True)
    # Determine number of available CPU cores
    n_cpus = os.cpu_count() or 1   # may return None
    NUM_PROCS = max(1, n_cpus - 1) # leave 1 core free
    print(f"Using {NUM_PROCS} out of {n_cpus} available CPU cores for multiprocessing.")

    for enc_type in encoding_types:
        print("Concatenating for encoding type:", enc_type)
        root_data = root_in+f'{model_type}_results_{out_name}/{enc_type}_data/out/'
        df = concatenate_dfs(root_data,filter_type, NUM_PROCS=NUM_PROCS, do_stats=do_stats)
        df.to_csv(root_out+f'/{model_type}_{enc_type}_{out_name}{additional}.csv',compression='gzip')
        print(f"Done saving {model_type}_{enc_type}_{out_name}{additional}.csv")
    print("All done!")

def main():
    parser = argparse.ArgumentParser(description='Concatenate CSV files.')
    parser.add_argument('--model_type', '-m', type=str, help='Model type, i.e. onehot')
    parser.add_argument('--out_name', '-o', type=str, help='Output name: loss + any additions, i.e. bce_batch9_ui')
    parser.add_argument('--filter_type', '-f', type=str, default=None, help='Filter type, i.e. acoustic_vec')
    parser.add_argument('--encoding_type', '-e', type=str, default='bigram,unigram,zerovec-bigram', help='Encoding type')
    parser.add_argument('--out_root', '-or', type=str, help='Root directory for output data', 
                        default='/projects/jurovlab/stat_learning/results/')
    parser.add_argument('--in_root', '-ir', type=str, help='Root directory for input data', 
                        default='/projects/jurovlab/stat_learning/interim/') 
    parser.add_argument('--do_stats', '-ds', action='store_true', help='Whether to compute statistics or not')
    args = parser.parse_args()
    print(args)
    process_dfs(args)

if __name__ == '__main__':
    main()



# id_cols = ['model_type','model_num','exp',
    #         'condition','epochs',
    #         # 'in_label','out_label',
    #         'batch_id',
    #         'btc_ep',
    #         'stim_type']

# def get_stat_point(df,cols):
#     df_stat = df.groupby(cols).agg(avg_loss=("loss", "mean"),
#                                             min_loss=("loss", "min"),
#                                             max_loss=("loss", "max"),
#                                             q1_loss=("loss", lambda x: x.quantile(0.25)),
#                                             q3_loss=("loss", lambda x: x.quantile(0.75))
#                                             ).reset_index()
#     return df_stat



# def concatenate_dfs(root, filter_type, id_cols = id_cols):
#     all_dfs = []
#     for file in os.listdir(root):
#         if file.endswith('.csv') and not file.startswith('loss') and (filter_type is None or filter_type in file):
#             print("Processing file:", file)
#             if os.path.getsize(os.path.join(root, file)) > 0:
#                 try:
#                     df = pd.read_csv(os.path.join(root, file), compression='gzip')
#                 except Exception as e:
#                     df = pd.read_csv(os.path.join(root, file))
#                 df['condition'] = df['condition'].map({1: 'Word', 0: 'Part-Word'})
#                 df['btc_ep']=df['batch_id']+(df['epochs']-1)*809
#                 df_stat = get_stat_point(df,id_cols)
#                 all_dfs.append(df_stat)
#     return pd.concat(all_dfs)