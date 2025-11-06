import torch; torch.manual_seed(108)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import  DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import librosa
import os
from ae_model import *
import sys
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
import seaborn as sns
from utils import *

# try:
#     norm = bool(int(sys.argv[1]))
#     print("Normalizing acoustic data")
# except IndexError:
#     norm = False
#     print("NOT Normalizing acoustic data")


loss_type = "mse"  #sys.argv[1] # bce, mse
hidden_fixed=True # whether first hidden size is fixed to 512

#####====================================#####

if hidden_fixed==True:
    # Changing bottleneck size
    N=1
    bottleneck_sizes = [8,16,32,64,128,256]
    num_layers=[N]
    hidden_size=512
    type_test_dir=f'changing_bottleneck_width_{N}'
else:
    # Changing layer size
    bottleneck_sizes = [8]
    num_layers=[1,2,3,4,5,6,7]
    type_test_dir='changing_layer_numbers'


batch_sizes = [1,2,4,8,16,32,64,128]
LRs = [0.0001,0.01, 0.00001, 0.001]
epochs=1
exps = [1,2] 
# model_types = ['onehot','acoustic','phon','acoustic_norm','acoustic_new','acoustic_new_norm']
model_types = ['onehot','phon','acoustic_new_norm']
out_root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/'
dfs=[]

print('Starting testing')
for model_type in model_types:
    print(f'Starting {model_type} models')
    for bottleneck_size in bottleneck_sizes:
        for num_layer in num_layers:
            if hidden_fixed == False:
                # if bottleneck_size=8, log2 of that is 3, so that we have hidden size 
                # that ends with bottleneck size but begins at an appropriate number
                # to be able to halve each layer (divide with 2) from starting hidden size
                # -1 to ensure that we start with hidden size that isn't twice too big
                # [difference between index and actual num of layers]
                hidden_size = int(2**(num_layer-1+np.log2(bottleneck_size))) 
            for exp in exps:
                if exp==1:
                    root = out_root+'simulation_one/unigram_data/'
                elif exp==2:
                    root = out_root+'simulation_two/unigram_data/'

                out_dir = out_root+f'models_testing_params/{type_test_dir}'
                os.makedirs(out_dir,exist_ok=True)
                os.makedirs(out_dir+f'/{model_type}_{bottleneck_size}_{num_layer}',exist_ok=True)

                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

                train_data = torch.load(root+f'{model_type}_train_in.pt',weights_only=True)
                seq_syllables = torch.load(root+f'{model_type}_train_out.pt',weights_only=True)
                syll_vec_ = np.load(root+f'syll_vec_{model_type}.npz', allow_pickle=True)

                syll_vec = {}
                for key in syll_vec_.keys():
                    syll_vec[key] = torch.tensor(syll_vec_[key].astype(float))

                train_dataset = TensorDataset(train_data)
                seq_dataset = TensorDataset(seq_syllables)

            
                for batch_size in batch_sizes:
                    dataloader_in = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
                    dataloader_out = DataLoader(seq_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
                    train_out = list(dataloader_out)
                    for LR in LRs:
                        for NUM in range(0,5):
                            # loss_track = []
                            eval_track = []
                            begin_track = []

                            if loss_type=='mse':
                                loss_fun = torch.nn.MSELoss() 
                                sigmoid=False    
                            elif loss_type=='bce':
                                loss_fun = torch.nn.BCELoss()
                                sigmoid=True
                            model = AE(input_shape=train_in.shape[1:],
                                        output_shape=seq_syllables.shape[1:],
                                        hidden_size=hidden_size,
                                        num_layers=num_layer,
                                        bottleneck_size=bottleneck_size, 
                                        sigmoid=sigmoid).to(device)
                            optimizer = torch.optim.Adam(model.parameters(),lr=LR)   
                            # test initialized model
                            for i, batch in enumerate(dataloader_in):
                                batch_out = train_out[i][0].to(device)
                                batch = batch[0].to(device)
                                codes, decoded = model(batch)
                                loss = loss_fun(decoded,batch_out)
                                begin_track.append(float(loss)) 
                            begin_df = pd.DataFrame(begin_track,columns=['loss'])
                            # train model
                            for epoch in range(1,epochs+1):
                                for i, batch in enumerate(dataloader_in):
                                    batch_out = train_out[i][0].to(device)
                                    batch = batch[0].to(device)
                                    codes, decoded = model(batch)
                                    optimizer.zero_grad()
                                    loss = loss_fun(decoded,batch_out)
                                    # loss_track.append(float(loss))
                                    loss.backward()
                                    optimizer.step()
                                
                                # torch.save(model.state_dict(), root+out_dir+f'/{model_type}/ckpt_ep-{epoch}_exp-{exp}_n-{NUM}_lr-{LR}_btc-{batch_size}_adam.pt')
                                
                                # test after 1 epoch
                                model.eval()
                                for i, batch in enumerate(dataloader_in):
                                    batch_out = train_out[i][0].to(device)
                                    batch = batch[0].to(device)
                                    codes, decoded = model(batch)
                                    loss = loss_fun(decoded,batch_out)
                                    eval_track.append(float(loss))
                                eval_df = pd.DataFrame(eval_track,columns=['loss'])
                                eval_df['difference'] = begin_df['loss'] - eval_df['loss']
                                eval_df['begin_loss'] = begin_df['loss']
                                eval_df.to_csv(out_dir+f'/{model_type}_{bottleneck_size}_{num_layer}/eval_ep-{epoch}_exp-{exp}_n-{NUM}_lr-{LR}_btc-{batch_size}.csv')
                                model.train()
                            dfs.append(eval_df)
                            print('done model #: ',NUM)
                        print('done learning rate models: ',LR)
                    print('done batch size: ',batch_size)
                print('done experiment: ',exp)
            print('done models with # layers: ', num_layer)
        print('done bottleneck_size #',bottleneck_size) 
    print('done model type: ',model_type)

pd.concat(dfs).to_csv(out_dir+'param_results.csv')
