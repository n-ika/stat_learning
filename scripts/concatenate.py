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
    root = args.root
    root_out = root+args.out_dir
    root_in = root+args.in_dir
    do_stats = args.do_stats
    additional = args.additional
    if filter_type is not None and additional=='':
        additional = f'_{filter_type}'

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
    parser.add_argument('--model_type', '-m', type=str, help='Model type, i.e. rnn/ae')
    parser.add_argument('--out_name', '-o', type=str, help='Output name: loss + any additions, i.e. bce_batch9_ui')
    parser.add_argument('--filter_type', '-f', type=str, default=None, help='Filter type, i.e. acoustic_vec')
    parser.add_argument('--encoding_type', '-e', type=str, default='bigram,unigram,zerovec-bigram', help='Encoding type')
    parser.add_argument('--root', '-r', type=str, help='Root directory for output data', 
                        default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--out_dir', '-od', type=str, help='Root directory for output data', 
                        default='results/')
    parser.add_argument('--in_dir', '-id', type=str, help='Root directory for input data', 
                        default='interim/') 
    parser.add_argument('--do_stats', '-ds', action='store_true', help='Whether to compute statistics or not')
    parser.add_argument('--additional', '-a', type=str, default='', help='Additional string to append to output filename')
    args = parser.parse_args()
    print(args)
    process_dfs(args)

if __name__ == '__main__':
    main()


