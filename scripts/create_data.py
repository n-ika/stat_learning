import torch; torch.manual_seed(108)
import torch.nn.functional as F
import os 
import numpy as np
import pandas as pd
import librosa
import argparse
# from models import *
from data_utils import *

def main(args):
    root = args.root
    out_dir = args.out_dir
    n_mels = args.n_mels
    for stim_type in ['unigram','bigram','zerovec-bigram']:
        for model_type in ['onehot', 'phon', f'acoustic_{n_mels}_norm']: 
            for exp in [1,2]:
                if exp==1:
                    exp_dir = 'simulation_one/'
                    test_items={'pa-bi-ku':1,'ti-bu-do':1,'tu-da-ro':0,'pi-go-la':0}
                    # train_items = {'pa-bi-ku': 1, 'ti-bu-do': 1, 'go-la-tu': 1, 'da-ro-pi': 1}
                elif exp==2:
                    exp_dir = 'simulation_two/'
                    # train_items={'gu-lo-ba':1,'bo-ga-ki':1,'pi-du-go':1,'ge-bi-tu':1}
                    test_items={'gu-lo-ba':1,'bo-ga-ki':1,'tu-pi-du':0,'go-ge-bi':0}
                root_exp = root + f'/data/'
                root_raw_stims = root + f'/data/' + exp_dir + 'original/'
                # out_dir = root + exp_dir + f"{stim_type}_data/"
                out_dir = root + f'/data/' + exp_dir + f"{stim_type}_data/"
                os.makedirs(out_dir, exist_ok=True)

                syll_vec = get_syll_vec(model_type,root_raw_stims,exp,n_mels)
                if model_type.endswith('norm'):
                    all_values = np.concatenate([v.flatten() for v in syll_vec.values()])
                    tr_min = np.min(all_values)
                    tr_max = np.max(all_values)
                    for syllv in syll_vec.keys():
                        syll_vec[syllv] = (syll_vec[syllv] - tr_min) / (tr_max - tr_min + 1e-6)
                transcript_df = pd.read_csv(root_exp +f'transcript_exp{exp}.csv',header=None)
                transcript_df.columns = ['stimulus']
                train_in, train_labels_in, train_out, train_labels_out, train_conditions = create_train_data(transcript_df,syll_vec,stim_type)
                test_in, test_labels_in, test_out, test_labels_out, test_conditions = create_test_data(test_items,syll_vec,stim_type)
                print(f'{stim_type} data, {model_type} model, exp: {exp}, train in shape: {train_in.shape}, train out shape: {train_out.shape}, test in shape: {test_in.shape}, test out shape: {test_out.shape}')
                torch.save(syll_vec, out_dir+f'syll_vec_{model_type}_{stim_type}.pt')
                save_data('train_in',train_in,train_labels_in,train_conditions,out_dir,model_type)
                save_data('train_out',train_out,train_labels_out,train_conditions,out_dir,model_type)
                save_data('test_in',test_in,test_labels_in,test_conditions,out_dir,model_type)
                save_data('test_out',test_out,test_labels_out, test_conditions, out_dir, model_type)
                if model_type.startswith('acoustic') and stim_type=='unigram':
                    for N in [1]:
                        syll_vec = {k:s.reshape(s.shape[0], s.shape[1], n_mels, -1) for k,s in syll_vec.items()}
                        torch.save(syll_vec, out_dir+f'syll_vec_acoustic_vec_{n_mels}_{stim_type}.pt')
                        train_in = train_in.reshape(-1, N, n_mels)
                        train_labels_in = np.repeat(train_labels_in, 28//N).tolist()
                        train_out = train_out.reshape(-1, N, n_mels)
                        train_labels_out = np.repeat(train_labels_out, 28//N).tolist()
                        test_in = test_in.reshape(-1, N, n_mels)
                        test_labels_in = np.repeat(test_labels_in, 28//N).tolist()
                        test_out = test_out.reshape(-1, N, n_mels)
                        test_labels_out = np.repeat(test_labels_out, 28//N).tolist()
                        train_conditions = np.repeat(train_conditions, 28//N).tolist()
                        test_conditions = np.repeat(test_conditions, 28//N).tolist()
                        save_data('train_in',train_in,train_labels_in,train_conditions,out_dir,f'acoustic_vec_{n_mels}')
                        save_data('train_out',train_out,train_labels_out,train_conditions,out_dir,f'acoustic_vec_{n_mels}')
                        save_data('test_in',test_in,test_labels_in,test_conditions,out_dir,f'acoustic_vec_{n_mels}')
                        save_data('test_out',test_out,test_labels_out, test_conditions, out_dir, f'acoustic_vec_{n_mels}')
                        print(f'{stim_type} data, acoustic_vec_{n_mels} model, exp: {exp}, train in shape: {train_in.shape}, train out shape: {train_out.shape}, test in shape: {test_in.shape}, test out shape: {test_out.shape}')
    print('data saved!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--out_dir', '-od', type=str, default='data/')
    parser.add_argument('--n_mels', type=int, default=39)
    args = parser.parse_args()
    main(args)
   
# tp_df.to_csv(root+f'tp_df_exp{exp}.csv')
# TODO