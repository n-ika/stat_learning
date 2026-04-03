from collections import defaultdict
import numpy as np
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from multiprocessing import Pool
from functools import partial


id_cols = [
            'condition',
            'btc_ep',
            ]

def get_stat_point(df,cols):
    df_stat = df.groupby(cols).agg(median=("loss", "median"),
                                    min=("loss", "min"),
                                    max=("loss", "max"),
                                    q1=("loss", lambda x: x.quantile(0.25)),
                                    q3=("loss", lambda x: x.quantile(0.75))
                                    ).reset_index()
    return df_stat

def interval_coded_BY_corrected_sign_test(data, alpha=0.01):
    # - Sign test for differences between word and part-word error with Benjamini Yekutieli FDR control 
    #   for dependent tests at level alpha (correction for multiple comparisons) and interval coding of 
    #   the output
    # - input :
    #   - 'data' should contain data for only one xp and one model type
    #   - 'data' should contain data for the different models (as per 'model_num') for the two conditions ('condition')
    #     and all the 'btc_ep' we want
    #   - if there are multiple entries for a given model_num + condition + btc_ep in 'data'
    #     (e.g. loss for each of four different test syllables), the corresponding losses will be averaged before
    #     any statistical testing is performed
    # - for the output interval coding of significance regions
    #   - both ends of each interval should be included 
    #     (i.e. they correspond to significant differences)
    #   - an interval of length 1, for example at position i, is coded as [i, i]
    data = data.groupby(['model_num', 'condition', 'btc_ep'], as_index=False)['loss'].mean()
    data = data.pivot(columns='condition', index=['model_num', 'btc_ep'], values=['loss'])
    data['loss_difference'] = data[('loss', 0)]-data[('loss', 1)]
    data = data.reset_index()
    p_values = data.groupby('btc_ep', as_index=False)['loss_difference'].apply(get_p_value)
    p_values = p_values.rename(columns={'loss_difference': 'p-value'})
    kept = apply_BY_correction(p_values, alpha=alpha)
    kept = kept.sort_values('btc_ep')
    intervals_with_significant_differences_in_median = interval_coding(kept)
    return intervals_with_significant_differences_in_median


def get_p_value(loss_differences):
    # function to be applied to one batch of 70 differences
    Nplus = len(loss_differences[loss_differences > 0])
    Nminus = len(loss_differences[loss_differences < 0])
    # for two-tailed test (for right-tailed, we'd take Nplus), taking the min is conservative 
    M = min(Nplus, Nminus)  
    # equalities are removed from degrees of freedom and we multiply by two for two-tailed
    p = 2*binom.cdf(k=M, n=Nplus+Nminus, p=0.5)
    return p

def apply_BY_correction(df, alpha=0.01):
    # Do Benjamini Yekutieli FDR control at .05
    p_data = df.sort_values('p-value')
    nb_comp = len(p_data)
    seq = np.arange(1, nb_comp+1).astype(float)
    seq_inv = 1./seq
    BY_seq = seq/float(nb_comp)*alpha/np.sum(seq_inv)
    inds = np.where(p_data['p-value'] > BY_seq)[0]
    if len(inds) == 0:
        kept = p_data
    else:
        ind0 = inds[0]
        kept = p_data.iloc[:ind0]
    return kept

def interval_coding(kept):
    kept = kept.sort_values('btc_ep')
    ix = np.where(kept['btc_ep'].diff() > 1)
    ends = kept['btc_ep'].iloc[ix[0]-1].values
    starts = kept['btc_ep'].iloc[ix[0]]
    ends = np.concat([ends, kept['btc_ep'].iloc[-1:]])
    starts =  np.concat([kept['btc_ep'].iloc[:1], starts])
    # interval coding
    # both ends of the interval should be included
    intervals_with_significant_differences_in_median = np.array(list(zip(starts, ends)))
    return intervals_with_significant_differences_in_median



def process_file(root, file):
    path = os.path.join(root, file)
    try:
        df = pd.read_csv(path, compression="gzip")
    except Exception:
        df = pd.read_csv(path)
    df['btc_ep'] = [0 if epoch==0 else (epoch-1)*809*28+batch+1 for batch, epoch in zip(df['batch_id'], df['epochs'])]
    df.drop(columns=['lr','in_label','out_label','stim_type','model_type','batch_id','epochs','exp'], inplace=True) 
    return df.reset_index(drop=True)

def concatenate_dfs(root, NUM_PROCS, exp):
    files = [
        f for f in os.listdir(root)
        if f.endswith(".csv")
        and not f.startswith("loss")
        and (f'exp-{exp}' in f)
    ]
    with Pool(processes=NUM_PROCS) as pool:
        dfs = pool.map(partial(process_file, root), files)
    return pd.concat(dfs, ignore_index=True)



def plot_stats(df_exp, exp, stim_structure, model_type, type_loss, intervals_with_significant_differences_in_median, out_root, logy):
    label_fontsize = 24
    model_name = model_type.split('_')[0]
    if model_name == 'phon':
        encoding_type = 'Phon. Enc.'
    elif model_name.startswith('acoustic'):
        encoding_type = 'Acoustic Enc.'
    elif model_name == 'onehot':
        encoding_type = 'Categorical Enc.'
    stim_type_readable = stim_structure.capitalize()
    if '9' in type_loss:
        btc = 9
    else:
        btc = 1
    fig, axes = plt.subplots(figsize=(10, 5), sharey=False)
    group_cols = ['condition']
    for keys, sub in df_exp.groupby(group_cols):
        sub = sub.sort_values("btc_ep")
        # readable label
        condition = keys[0]
        label = f"{'Word' if condition == 1 else 'Part Word'}"
        axes.plot(sub["btc_ep"], sub["median"], label=label, linewidth=3)
        
        # band between Q1 and Q3 if present
        if ('q1' in sub.columns) and ('q3' in sub.columns):
            axes.fill_between(
                sub["btc_ep"],
                sub["q1"],
                sub["q3"],
                alpha=0.2,
                color=axes.get_lines()[-1].get_color(),
                label=f"{label} IQR"
            )

        # band between min_loss and max_loss if present
        if ('min' in sub.columns) and ('max' in sub.columns):
            axes.fill_between(
                sub["btc_ep"],
                sub["min"],
                sub["max"],
                alpha=0.1,
                color=axes.get_lines()[-1].get_color(),
                label=f"{label} Min-Max"
            )
    # plotting the results
    print('plotting stats')
    for interval in intervals_with_significant_differences_in_median:
        axes.plot(interval, [1,1], '-k', linewidth=5, label='Paired sign test significance' if interval[0]==intervals_with_significant_differences_in_median[0][0] else "")
    plt.grid()

    axes.set_xlabel("Training step", fontsize=label_fontsize)
    axes.set_ylabel("Loss value", fontsize=label_fontsize)
    # axes.legend(ncol=1, fontsize=label_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axes.set_title(f"{stim_type_readable} | {encoding_type} | Btc. = {btc} | Exp. {exp}", fontsize=label_fontsize+2, pad=20) 
    if logy:
        axes.set_yscale('log')
    axes.set_ylabel("Loss value", fontsize=label_fontsize)
    # fig.suptitle(f"{stim_type_readable} | {encoding_type}", fontsize=label_fontsize)
    ticks = [0.01, 0.1, 1] #1e-5, 1e-4, 1e-3, 
    axes.set_yticks(ticks)
    epochs = [1,2,4,8,16]
    lines_at_x = [int(df_exp['btc_ep'].max()/x) for x in epochs]
    for x in lines_at_x:
        plt.axvline(x=x, color='k', linestyle='--', ymin=0, ymax=1)
    plt.tick_params(axis='both', which='major', labelsize=label_fontsize-2)
    plt.tight_layout()
    plt.savefig(out_root+f"{model_type}_{stim_structure}_{type_loss}_{exp}{'_logy' if logy else ''}.pdf")
    # plt.ylim(0, 1)
    plt.show()
    plt.close()
    return(df_exp,intervals_with_significant_differences_in_median)


def process_dfs(args):
    architecture = args.architecture
    type_loss = args.type_loss
    encoding_type = args.encoding_type
    stim_structure = args.stim_structure
    root = args.root
    root_out = root+args.out_dir
    root_in = root+args.in_dir
    logy = args.logy
    additional = args.additional
    exp_n = args.exp_n
    fig_path = root_out+f'/figures/'
    os.makedirs(root_out, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    # Determine number of available CPU cores
    n_cpus = os.cpu_count() or 1   # may return None
    NUM_PROCS = max(1, n_cpus - 1) # leave 1 core free
    print(f"Using {NUM_PROCS} out of {n_cpus} available CPU cores for multiprocessing.")

    print(f"Concatenating for encoding type {encoding_type}, stim structure {stim_structure}, exp #{exp_n}")
    root_data = root_in+f'/{architecture}_results_{type_loss}/{stim_structure}_data/out/'
    df = concatenate_dfs(root_data, NUM_PROCS=NUM_PROCS, exp=exp_n)
    df.to_csv(root_out+f'/{architecture}_{stim_structure}_{type_loss}{additional}_{exp_n}.csv',compression='gzip')
    print(f"Done saving full df")
    
    intervals_with_significant_differences_in_median = interval_coded_BY_corrected_sign_test(df)
    np.save(root_out+f'/intervals_{architecture}_{stim_structure}_{type_loss}{additional}_{exp_n}.npy', intervals_with_significant_differences_in_median)
    print(f"Done saving intervals")
    
    df = get_stat_point(df,id_cols)
    df.to_csv(root_out+f'/stat_{architecture}_{stim_structure}_{type_loss}{additional}_{exp_n}.csv',compression='gzip')
    print(f"Done saving stat df")

    plot_stats(df, exp_n, stim_structure, type_loss, intervals_with_significant_differences_in_median, fig_path, logy, additional)

    print("All done!")

def main():
    parser = argparse.ArgumentParser(description='Concatenate CSV files.')
    parser.add_argument('--architecture', '-a', type=str, help='Model type, i.e. rnn/ae')
    parser.add_argument('--type_loss', '-tl', type=str, help='Output name: loss + any additions, i.e. bce_batch9_ui')
    parser.add_argument('--root', '-r', type=str, help='Root directory for output data', 
                        default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--out_dir', '-od', type=str, help='Root directory for output data', 
                        default='results/')
    parser.add_argument('--in_dir', '-id', type=str, help='Root directory for input data', 
                        default='interim/') 
    # parser.add_argument('--do_stats', '-ds', action='store_true', help='Whether to compute statistics or not')
    parser.add_argument('--additional', '-ad', type=str, default='', help='Additional string to append to output filename')
    parser.add_argument('--encoding_type', '-et', type=str, default='acoustic_vec_16', 
                        help='Encoding types to use, i.e. unigram, bigram, zerobigram')
    parser.add_argument('--stim_structure', '-st', type=str, default='unigram', 
                        help='Stimulus structure to use, i.e. unigram')
    parser.add_argument('--exp_n', '-en', type=int, required=True, default=1, help='Experiment number, 1 or 2')
    parser.add_argument('--logy', '-ly', action='store_true', help='Whether to use log scale for y axis')
    args = parser.parse_args()
    print(args)
    process_dfs(args)

if __name__ == '__main__':
    main()


