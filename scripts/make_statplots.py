import numpy as np
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import argparse
import os

# main code

id_cols = ['model_type','exp',
            'condition','epochs',
            # 'in_label','out_label',
            'batch_id',
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


def plot_stats(df_exp, exp, stim_structure, type_loss, intervals_with_significant_differences_in_median, out_root):
    fig, axes = plt.subplots(figsize=(15, 5), sharey=False)
    group_cols = ['model_type', 'condition']
    for keys, sub in df_exp.groupby(group_cols):
        sub = sub.sort_values("btc_ep")
        # readable label
        model_type, condition = keys
        label = f"{condition.capitalize()}"
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

    # axes.set_yscale('log')

    axes.set_ylabel("Loss value")
    fig.suptitle(f"{df_exp.iloc[0]['stim_type'].capitalize()} | {model_type.split('_')[0].capitalize()}: Median, IQR, and full range over training")
    plt.tight_layout()
    plt.savefig(out_root+f"{model_type}_{stim_structure}_{type_loss}_{exp}.svg")
    plt.show()
    plt.close()


def load_and_process_data(args):
    in_root = args.in_root
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    model_architecture = args.architecture
    type_loss = args.loss_type
    encoding_types = args.encoding_types 
    stim_structures = args.stim_structure
    
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
                if df_full.batch_id.unique().shape[0]>500:
                    df_full['btc_ep'] = [0 if epoch==0 else (epoch-1)*809+batch+1 for batch, epoch in zip(df_full['batch_id'], df_full['epochs'])]
                else:
                    df_full['btc_ep'] = df_full['epochs']

                # selecting data for one xp and one model type
                test_data = df_full[(df_full['model_type'] == encoding_type) & (df_full['exp'] == exp)]
                
                # running statistical tests
                print('running tests')
                intervals_with_significant_differences_in_median = interval_coded_BY_corrected_sign_test(test_data)
                df_exp = get_stat_point(test_data,id_cols)
                plot_stats(df_exp, exp, stim_structure, type_loss, intervals_with_significant_differences_in_median, out_root)


##################
# def interval_coded_BY_corrected_sign_test(data, alpha=0.01):
#     # - Sign test for differences between word and part-word error with Benjamini Yekutieli FDR control 
#     #   for dependent tests at level alpha (correction for multiple comparisons) and interval coding of 
#     #   the output
#     # - input :
#     #   - 'data' should contain data for only one xp and one model type
#     #   - 'data' should contain data for the different models (as per 'model_num') for the two conditions ('condition')
#     #     and all the 'btc_ep' we want
#     #   - if there are multiple entries for a given model_num + condition + btc_ep in 'data'
#     #     (e.g. loss for each of four different test syllables), the corresponding losses will be averaged before
#     #     any statistical testing is performed
#     # - for the output interval coding of significance regions
#     #   - both ends of each interval should be included 
#     #     (i.e. they correspond to significant differences)
#     #   - an interval of length 1, for example at position i, is coded as [i, i]
#     data = data.groupby(['model_num', 'condition', 'btc_ep'], as_index=False)['loss'].mean()
#     data = data.pivot(columns='condition', index=['model_num', 'btc_ep'], values=['loss'])
#     data['loss_difference'] = data[('loss', 'Part-Word')]-data[('loss', 'Word')]
#     data = data.reset_index()
#     p_values = data.groupby('btc_ep', as_index=False)['loss_difference'].apply(get_p_value)
#     p_values = p_values.rename(columns={'loss_difference': 'p-value'})
#     kept = apply_BY_correction(p_values, alpha=alpha)
#     kept = kept.sort_values('btc_ep')
#     intervals_with_significant_differences_in_median = interval_coding(kept)
#     return intervals_with_significant_differences_in_median


# def get_p_value(loss_differences):
#     # function to be applied to one batch of 70 differences
#     Nplus = len(loss_differences[loss_differences > 0])
#     Nminus = len(loss_differences[loss_differences < 0])
#     # for two-tailed test (for right-tailed, we'd take Nplus), taking the min is conservative 
#     M = min(Nplus, Nminus)  
#     # equalities are removed from degrees of freedom and we multiply by two for two-tailed
#     p = 2*binom.cdf(k=M, n=Nplus+Nminus, p=0.5)
#     return p


# def apply_BY_correction(df, alpha=0.01):
#     # controlling for multiple comparisons (Benjamini Yekutieli FDR control for dependent tests)
#     # Do Benjamini Yekutieli FDR control at .05
#     p_data = df.sort_values('p-value')
#     nb_comp = len(p_data)
#     seq = np.arange(1, nb_comp+1).astype(float)
#     seq_inv = 1./seq
#     BY_seq = seq/float(nb_comp)*alpha/np.sum(seq_inv)
#     inds = np.where(p_data['p-value'] > BY_seq)[0]
#     if len(inds) == 0:
#         kept = p_data
#     else:
#         ind0 = inds[0]
#         kept = p_data.iloc[:ind0]
#     return kept


# def interval_coding(kept):
#     kept = kept.sort_values('btc_ep')
#     ix = np.where(kept['btc_ep'].diff() > 1)
#     ends = kept['btc_ep'].iloc[ix[0]-1].values
#     starts = kept['btc_ep'].iloc[ix[0]]
#     ends = np.concat([ends, kept['btc_ep'].iloc[-1:]])
#     starts =  np.concat([kept['btc_ep'].iloc[:1], starts])
#     # interval coding
#     # both ends of the interval should be included
#     intervals_with_significant_differences_in_median = np.array(list(zip(starts, ends)))
#     return intervals_with_significant_differences_in_median

# def get_stat_point(df,cols):
#     df_stat = df.groupby(cols).agg(median=("loss", "median"),
#                                     min=("loss", "min"),
#                                     max=("loss", "max"),
#                                     q1=("loss", lambda x: x.quantile(0.25)),
#                                     q3=("loss", lambda x: x.quantile(0.75))
#                                     ).reset_index()
#     return df_stat


# def create_plot(args):
#     encoding_types = args.encoding_types
#     root_in = args.in_root
#     root_out = args.out_root
#     architecture = args.architecture
#     type_loss = args.loss_type
#     stim_structures = args.stim_structure
#     logy = args.logy
#     os.makedirs(root_out, exist_ok=True)
#     id_cols = ['model_type','exp',
#             'condition','epochs',
#             # 'in_label','out_label',
#             'batch_id',
#             'btc_ep',
#             'stim_type']
#     for encoding_type in encoding_types:
#         for stim_structure in stim_structures:
#             for exp in [1, 2]:
#                 print(f'Analyzing: {architecture} - {encoding_type} - {stim_structure}, experiment: {exp}')
#                 try:
#                     df = pd.read_csv(root_in+f'{architecture}_{stim_structure}_{type_loss}.csv',compression='gzip')
#                 except Exception:
#                     print('file not found, skipping ', root_in+f'{architecture}_{stim_structure}_{type_loss}.csv')
#                     pass
#                 # loading and formatting data
#                 print('loading data')
#                 del df['Unnamed: 0']
#                 df = df.reset_index()
#                 del df['index']
#                 df['btc_ep'] = [0 if epoch==0 else (epoch-1)*809+batch+1 for batch, epoch in zip(df['batch_id'], df['epochs'])]

#                 # selecting data for one xp and one model type
#                 test_data = df[(df['model_type'] == encoding_type) & (df['exp'] == exp)]
#                 # running statistical tests
#                 print('running tests')
#                 intervals_with_significant_differences_in_median = interval_coded_BY_corrected_sign_test(test_data)
#                 # Get 5 points summary statistics for the df
#                 df_stat = get_stat_point(df,id_cols)
#                 df_exp = df_stat[df_stat['exp'] == exp]

#                 # plotting the results
#                 print('plotting')
#                 fig, axes = plt.subplots(figsize=(15, 5), sharey=False)
#                 group_cols = ['model_type', 'condition']
#                 for keys, sub in df_exp.groupby(group_cols):
#                     sub = sub.sort_values("btc_ep")
#                     # readable label
#                     model_type, condition = keys
#                     label = f"{condition.capitalize()}"
#                     axes.plot(sub["btc_ep"], sub["median"], label=label)
                    
#                     # band between Q1 and Q3 if present
#                     if ('q1' in sub.columns) and ('q3' in sub.columns):
#                         # remove invalid keyword 'color_label' and optionally add a label for the patch
#                         axes.fill_between(
#                             sub["btc_ep"],
#                             sub["q1"],
#                             sub["q3"],
#                             alpha=0.2,
#                             color=axes.get_lines()[-1].get_color(),
#                             label=f"{label} IQR"
#                         )

#                     # band between min_loss and max_loss if present
#                     if ('min' in sub.columns) and ('max' in sub.columns):
#                         axes.fill_between(
#                             sub["btc_ep"],
#                             sub["min"],
#                             sub["max"],
#                             alpha=0.1,
#                             color=axes.get_lines()[-1].get_color(),
#                             label=f"{label} Min-Max"
#                         )
#                 # plotting the results
#                 print('plotting stats')
#                 for interval in intervals_with_significant_differences_in_median:
#                     axes.plot(interval, [0,0], '-k', label='Significant difference, BY corrected')
#                 plt.grid()

#                 axes.set_xlabel("Training step")
#                 axes.set_ylabel("Loss value")
#                 axes.legend(fontsize='small', ncol=1)
#                 axes.set_title(f"Experiment {exp}")
#                 if logy:
#                     axes.set_yscale('log')
#                 axes.set_ylabel("Loss value")
#                 fig.suptitle(f"{df.iloc[0]['stim_type'].capitalize()} | {model_type.split('_')[0].capitalize()}: Median, IQR, and full range over training")
#                 plt.tight_layout()
#                 plt.savefig(root_out+f'{architecture}_{stim_structure}_{encoding_type}_{type_loss}_exp{exp}.png')
#                 plt.close()
#                 print(f"Figure saved to {root_out+f'{architecture}_{stim_structure}_{encoding_type}_{type_loss}_exp{exp}.png'}")


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of model results')
    parser.add_argument('--in_root', '-ir', type=str, help='Root directory for input data', 
                        default='/projects/jurovlab/stat_learning/results/')
    parser.add_argument('--out_root', '-or', type=str, help='Root directory for output data', 
                        default='/projects/jurovlab/stat_learning/results/figures/')                    
    parser.add_argument('--architecture', '-a', type=str, required=True, help='Type of model to analyze (e.g. rnn, ae)')
    parser.add_argument('--encoding_types', '-et', nargs='+', default=['onehot', 'phon', 'acoustic_new_norm'], 
                        help='List of encoding types to use')
    parser.add_argument('--stim_structure', '-st', nargs='+', default=['unigram', 'zerovec-bigram', 'bigram'], 
                        help='List of stimulus structure to use')
    parser.add_argument('--loss_type', '-lt', type=str, required=True, help='Type of loss function used (e.g. bce, mse, bce_batch9_ui)')
    parser.add_argument('--logy', action='store_true', help='Use logarithmic scale for y axis')
    args = parser.parse_args()
    print('Arguments:', args)
    load_and_process_data(args)

if __name__ == '__main__':
    main()