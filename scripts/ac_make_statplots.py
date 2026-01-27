from collections import defaultdict
import numpy as np
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

id_cols = ['exp',
            'condition',
            # 'in_label','out_label',
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


def plot_stats(df_exp, exp, stim_structure, model_type, type_loss, intervals_with_significant_differences_in_median, out_root, logy):
    fig, axes = plt.subplots(figsize=(15, 5), sharey=False)
    group_cols = ['condition']
    for keys, sub in df_exp.groupby(group_cols):
        sub = sub.sort_values("btc_ep")
        # readable label
        condition = keys[0]
        label = f"{'Word' if condition == 1 else 'Part Word'}"
        axes.plot(sub["btc_ep"], sub["median"], label=label)
        
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
        axes.plot(interval, [0,0], '-k')
    plt.grid()

    axes.set_xlabel("Training step")
    axes.set_ylabel("Loss value")
    axes.legend(fontsize='small', ncol=1)
    axes.set_title(f"Experiment {exp}")

    if logy:
        axes.set_yscale('log')
    axes.set_ylabel("Loss value")
    fig.suptitle(f"{df_exp.iloc[0]['stim_type'].capitalize()} | {model_type.split('_')[0].capitalize()}: Median, IQR, and full range over training")
    plt.tight_layout()
    plt.savefig(out_root+f"{model_type}_{stim_structure}_{type_loss}_{exp}{'_logy' if logy else ''}.svg")
    plt.close()


def load_and_process_data(args):
    in_root = args.in_root
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    model_architecture = args.architecture
    type_loss = args.loss_type
    encoding_type = args.encoding_type
    stim_structure = args.stim_structure
    logy = args.logy
    exp_n = args.exp_n
    base_dir =  in_root + f'/{model_architecture}_results_{type_loss}/{stim_structure}_data/out/'
    
    groups = {}
    print('loading paths of original dfs')
    for root, dirs, fnames in os.walk(base_dir):
        for fname in fnames:
            if not fname.endswith('.csv'):
                continue
            if 'exp-' not in fname or 'ep-' not in fname:
                continue
            if fname.startswith(encoding_type):
                # cheap way to get exp and ep substrings (you can parse numbers if needed)
                exp_part = next((s for s in fname.split('_') if s.startswith('exp-')), None)
                ep_part  = next((s for s in fname.split('_') if s.startswith('ep-')), None)

                if exp_part and ep_part:
                    key = (exp_part, ep_part)   # e.g. ('exp-2', 'ep-7')
                    path = os.path.join(root, fname)
                    groups.setdefault(key, []).append(path)
    
    print('loading original dfs')
    fin = []
    intervals = []
    for (exp, ep), paths in groups.items():
        if int(exp.split('-')[-1]) == exp_n:
            dfs = []
            print(f"Group {exp}, {ep}: {len(paths)} files")
            for path in paths:
                df = pd.read_csv(path, compression='gzip')
                df.drop(columns=['lr','in_label','out_label','stim_type','model_type'], inplace=True) 
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df['btc_ep'] = [0 if epoch==0 else (epoch-1)*809+batch+1 for batch, epoch in zip(combined_df['batch_id'], combined_df['epochs'])]
            intervals_with_significant_differences_in_median = interval_coded_BY_corrected_sign_test(combined_df)
            df_exp = get_stat_point(combined_df,id_cols)
            fin.append(df_exp)
            intervals.append(intervals_with_significant_differences_in_median)


    print('concatenating final stat df')
    dffull = pd.concat(fin, ignore_index=True).reset_index(drop=True)
    dffull['stim_type'] = stim_structure
    non_empty = [arr for arr in intervals if arr.size > 0]
    stacked = np.vstack(non_empty)
    sorted_arr = stacked[np.argsort(stacked[:, 0])]
    dffull.to_csv(out_root+f'data/summary_{model_architecture}_{stim_structure}_{type_loss}_exp{exp_n}.csv', index=False)
    np.save(out_root+f'data/intervals_{model_architecture}_{stim_structure}_{type_loss}_exp{exp_n}.npy', sorted_arr)
    print('plotting')
    plot_stats(dffull, exp, stim_structure, encoding_type, type_loss, sorted_arr, out_root, logy)
    print('done')


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of model results')
    parser.add_argument('--in_root', '-ir', type=str, help='Root directory for input data', 
                        default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--out_root', '-or', type=str, help='Root directory for output data', 
                        default='/projects/jurovlab/stat_learning/results/figures/')                    
    parser.add_argument('--architecture', '-a', type=str, required=True, help='Type of model to analyze (e.g. rnn, ae)')
    parser.add_argument('--encoding_type', '-et', type=str, default='acoustic_vec_1', 
                        help='Encoding types to use, i.e. unigram, bigram, zerobigram')
    parser.add_argument('--stim_structure', '-st', type=str, default='unigram', 
                        help='Stimulus structure to use, i.e. unigram')
    parser.add_argument('--loss_type', '-lt', type=str, required=True, help='Type of loss function used (e.g. bce, mse, bce_batch9_ui)')
    parser.add_argument('--logy', action='store_true', help='Use logarithmic scale for y axis')
    parser.add_argument('--exp_n', '-en', type=int, required=True, default=1, help='Experiment number, 1 or 2')
    args = parser.parse_args()
    print('Arguments:', args)
    load_and_process_data(args)

if __name__ == '__main__':
    main()

