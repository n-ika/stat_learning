import numpy as np
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import argparse
import os

# main code

id_cols = ['model_type',
           'exp',
            'condition',
            # 'epochs',
            # 'in_label','out_label',
            # 'batch_id',
            'btc_ep',
            'stim_type']

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
    data['loss_difference'] = data[('loss', 'Part-Word')]-data[('loss', 'Word')]
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


def plot_stats(df_exp, exp, stim_structure, type_loss, intervals_with_significant_differences_in_median, out_root, logy, additional):
    label_fontsize = 24
    fig, axes = plt.subplots(figsize=(10, 5), sharey=False)
    group_cols = ['model_type', 'condition']
    for keys, sub in df_exp.groupby(group_cols):
        sub = sub.sort_values("btc_ep")
        # readable label
        model_type, condition = keys
        model_name = model_type.split('_')[0]
        if model_name == 'phon':
            encoding_type = 'Phon. Encoding'
        elif model_name.startswith('acoustic'):
            encoding_type = 'Acoustic Encoding'
        elif model_name == 'onehot':
            encoding_type = 'Categorical Encoding'
        stim_type = df_exp.iloc[0]['stim_type']
        if stim_type == 'zerovec-bigram' or stim_type == 'zerobigram':
            stim_type_readable = 'Bigram with Pauses'
        else:
            stim_type_readable = stim_type.capitalize()
        label = f"{condition.capitalize()}"
        if '9' in type_loss:
            btc = 9
        else:
            btc = 1
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
        axes.plot(interval, [1,1], '-k', linewidth=5, label='Significant difference in median' if interval[0]==intervals_with_significant_differences_in_median[0][0] else "")
    plt.grid()
    axes.set_xlabel("Training step", fontsize=label_fontsize)
    axes.set_ylabel("Loss value", fontsize=label_fontsize)
    # axes.legend(ncol=1, fontsize=label_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axes.set_title(f"{stim_type_readable} | {encoding_type} | Btc. = {btc} | Exp. {exp}", fontsize=label_fontsize+2, pad=20) 
    if logy:
        axes.set_yscale('log')
    axes.set_ylabel("Loss value", fontsize=label_fontsize)
    # fig.suptitle(f"{stim_type_readable} | {encoding_type}", fontsize=label_fontsize)
    epochs = [1,2,4,8,16]
    lines_at_x = [int(df_exp['btc_ep'].max()/x) for x in epochs]
    for x in lines_at_x:
        plt.axvline(x=x, color='k', linestyle='--', ymin=0, ymax=1)
    ticks = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
    axes.set_yticks(ticks)
    plt.tick_params(axis='both', which='major', labelsize=label_fontsize-2)
    plt.tight_layout()
    plt.savefig(out_root+f"{model_type}_{stim_structure}_{type_loss}_{exp}{'_logy' if logy else ''}{additional}.svg")
    plt.close()


def load_and_process_data(args):
    root = args.root
    out_folder = args.out_folder + '/'
    out_root = root+out_folder+'figures/'
    in_root = root+out_folder
    os.makedirs(out_root, exist_ok=True)
    model_architecture = args.architecture
    type_loss = args.loss_type
    encoding_types = args.encoding_types 
    stim_structures = args.stim_structure
    logy = args.logy
    additional = args.additional
    
    for encoding_type in encoding_types:
        for stim_structure in stim_structures:
            for exp in [1, 2]:
                print(f'Analyzing: {model_architecture} - {encoding_type} - {stim_structure}, experiment: {exp}')
                df_full = pd.read_csv(in_root+f'{model_architecture}_{stim_structure}_{type_loss}.csv',compression='gzip')
                # loading and formatting data
                print('loading data')
                del df_full['Unnamed: 0']
                df_full = df_full.reset_index()
                del df_full['index']
                # df_full['btc_ep'] = df_full['epochs'].copy()
                # if df_full.batch_id.unique().shape[0]>500:
                max_val = df_full[df_full['epochs'] == 1]['batch_id'].max() + 1
                vals = df_full['batch_id'].values
                min_val = vals[vals > 0]
                step_size = int(min_val.min()) if min_val.size > 0 else 1
                df_full['btc_ep'] = [0 if epoch==0 else ((epoch-1)*max_val+batch+1)//step_size for batch, epoch in zip(df_full['batch_id'], df_full['epochs'])]
                #     df_full['btc_ep'] = [0 if epoch==0 else (epoch-1)*59+batch+1 for batch, epoch in zip(df_full['batch_id'], df_full['epochs'])]
                # selecting data for one xp and one model type
                test_data = df_full[(df_full['model_type'] == encoding_type) & (df_full['exp'] == exp)]
                
                #TODO
                # test_data = test_data[test_data['btc_ep']<=810]
                
                # running statistical tests
                print('running tests')
                intervals_with_significant_differences_in_median = interval_coded_BY_corrected_sign_test(test_data)
                df_exp = get_stat_point(test_data,id_cols)
                print('plotting')
                plot_stats(df_exp, exp, stim_structure, type_loss, intervals_with_significant_differences_in_median, out_root, logy, additional)

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of model results')
    parser.add_argument('--root', '-r', type=str, help='Root directory for input data', 
                        default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--out_folder', '-of', type=str, help='Root directory for output data', 
                        default='results')                    
    parser.add_argument('--architecture', '-a', type=str, required=True, help='Type of model to analyze (e.g. rnn, ae)')
    parser.add_argument('--encoding_types', '-et', nargs='+', default=['onehot', 'phon', 'acoustic_16_norm'], 
                        help='List of encoding types to use')
    parser.add_argument('--stim_structure', '-st', nargs='+', default=['unigram', 'zerovec-bigram', 'bigram'], 
                        help='List of stimulus structure to use')
    parser.add_argument('--loss_type', '-lt', type=str, required=True, help='Type of loss function used (e.g. bce, mse, bce_batch9_ui)')
    parser.add_argument('--logy', action='store_true', help='Use logarithmic scale for y axis')
    parser.add_argument('--additional', '-ad', default='', help='Additional descriptor for output files')
    args = parser.parse_args()
    print('Arguments:', args)
    load_and_process_data(args)
    print('Done!')

if __name__ == '__main__':
    main()