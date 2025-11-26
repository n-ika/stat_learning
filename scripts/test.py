import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

df = pd.read_csv('/projects/jurovlab/stat_learning/ae_results_mse/zerovec-bigram_data/out/acoustic_new_norm_8_exp-1_n-1_lr-0.001_train-1_test-1_Adam_mse_none_ep-0.csv')
df.to_csv('/scratch/jurovlab/testing_writing.csv', index=False, compression='gzip')