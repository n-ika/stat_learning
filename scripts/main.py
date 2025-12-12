import torch
import torch.nn as nn
from torch.utils.data import  DataLoader, TensorDataset
import numpy as np
import pandas as pd
import librosa
import os
import sys
from train_utils import *
from models import *
import argparse

def main(args):
    in_root = args.in_root
    out_root = args.out_root
    model_arch = args.model_arch
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    bottleneck_size = args.bottleneck_size
    hidden_size = args.hidden_size
    num_layer = args.num_layer
    epochs = args.epochs
    loss_type = args.loss_type
    lr = args.learning_rate
    # sigmoid = args.sigmoid
    num_simulations = args.num_simulations
    optional = args.optional
    # all_dfs = []

    if loss_type == 'mse':
        loss_fun = nn.MSELoss(reduction='mean')
        sigmoid_type = 'none'
        sigmoid = False
        print("Using MSE loss: last layer should not be sigmoid")
    elif loss_type == 'bce':
        loss_fun = nn.BCELoss()
        sigmoid_type = 'sigmoid'
        sigmoid = True
        print("Setting last layer to sigmoid layer due to BCE loss")
    else:
        raise ValueError("Invalid loss type. Choose 'mse' or 'bce'.")

    for model_type in ['acoustic_vec_1']:#, 'acoustic_new_norm', 'onehot', 'phon']: #TODO
        for stim_type in ['unigram']: #, 'zerovec-bigram', 'bigram']: #TODO
            input_type = stim_type + '_data'
            for exp in [1,2]:
                if exp==1:
                    root_exp = in_root+'simulation_one/'
                elif exp==2:
                    root_exp = in_root+'simulation_two/'
                if model_arch == 'AE':
                    root_dir = out_root+f'/ae_results_{loss_type}{optional}/' #FIXME
                elif model_arch == 'RNN':
                    root_dir = out_root+f'/rnn_results_{loss_type}{optional}/' #FIXME
                res_dir = root_dir+f'/{input_type}/out/'
                os.makedirs(res_dir,exist_ok=True)
                model_dir = root_dir+f'/{input_type}/models/exp_{exp}/{model_type}/' #_{bottleneck_size}_{num_layer}_{hidden_size}
                os.makedirs(model_dir,exist_ok=True)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print('Using device:', device)

                syll_vec,in_loader,out_loader,test_in_loader,test_out_loader = load_all_data(root_exp,
                                                                                             model_type,
                                                                                             input_type,
                                                                                             batch_size_train,
                                                                                             batch_size_test)
                test_out_list = list(test_out_loader)
                out_loader_list = list(out_loader)
                batch_sample = next(iter(in_loader))

                for simulation_num in range(1,num_simulations+1):
                    if model_arch == 'AE':
                        if model_type.startswith('acoustic_vec'):
                            out_size = [1,16] #TODO
                        else:
                            out_size = syll_vec['pi'].shape[1:]
                        model = AE(input_size=batch_sample[0].shape[1:],
                                   output_size=out_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layer,
                                   bottleneck_size=bottleneck_size,
                                   sigmoid=sigmoid).to(device)
                    elif model_arch == 'RNN':
                        model = RNNModel(input_size=batch_sample[0].shape[-1],
                                         output_size=np.prod(syll_vec['pi'].shape[1:]),
                                         hidden_size=hidden_size,                                         
                                         loss_type=loss_type).to(device)
                        # model = RNNModel(input_size=batch_sample[0].reshape(1,1,-1).shape[-1], #FIXME
                        #                  output_size=np.prod(syll_vec['pi'].shape[1:]),
                        #                  hidden_size=hidden_size,                                         
                        #                  loss_type=loss_type).to(device)
                    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

                    print(f'Creating model: {model_type}, lr: {lr}, experiment: {exp}, stimulus type: {stim_type},'
                          f'number: {simulation_num}, train batch: {batch_size_train}, test batch: {batch_size_test}')   
                    out_name = (
                            f"{model_type}_{bottleneck_size}_exp-{exp}_n-{simulation_num}_lr-{lr}_train-{int(batch_size_train)}_test"
                            f"-{int(batch_size_test)}_{optimizer.__class__.__name__}"
                            f"_{loss_type}_{sigmoid_type}"
                        )
                    # model_ckpt_name = model_dir+f'/ckpt_{out_name}'
                    train_model(model,
                                optimizer,
                                loss_fun,
                                device,
                                epochs,
                                in_loader,
                                out_loader_list,
                                test_in_loader,
                                test_out_list,
                                batch_size_test,
                                stim_type,
                                model_type,
                                lr,
                                exp,
                                simulation_num,
                                model_dir,
                                res_dir,
                                out_name)
                    # all_dfs.append(test_df)
                    print('done model #: ',simulation_num)
                print('done experiment: ',exp)
            print('done stimuli type:', stim_type)
        print('done model type:', model_type)
    # df_all = pd.concat(all_dfs)
    # df_all.to_csv(root_dir+f'/all_results_{model_arch}_{loss_type}_{sigmoid_type}.csv')
    print('done all!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', '-ir', type=str, help='Root directory for data', 
                        default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--out_root', '-or', type=str, help='Root directory for results', 
                        default='/projects/jurovlab/stat_learning/')
    parser.add_argument('--model_arch', '-ma', type=str, choices=['AE', 'RNN'], default='AE', help='Model architectures to run')
    parser.add_argument('--batch_size_train', '-btr', type=int, default=1, help='Training batch size')
    parser.add_argument('--batch_size_test', '-bte', type=int, default=1, help='Testing batch size')
    parser.add_argument('--bottleneck_size', '-bs', type=int, default=8, help='Size of the bottleneck layer')
    parser.add_argument('--hidden_size', '-hs', type=int, default=512, help='Size of the hidden layers')
    parser.add_argument('--num_layer', '-nl', type=int, default=7, help='Number of layers in the model')
    parser.add_argument('--epochs', '-e', type=int, default=16, help='Number of training epochs')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--loss_type', '-lt', type=str, choices=['mse', 'bce'], default='mse', help='Type of loss function')
    # parser.add_argument('--sigmoid', '-s', type=bool, default=False, help='Use sigmoid activation in the output layer')
    parser.add_argument('--num_simulations', '-ns', type=int, default=70, help='Number of simulations to run')
    parser.add_argument('--optional', '-o', type=str, default='', 
                        help='Optional out folder descriptor (output folder is [model type]_[loss type][optional])')
    args = parser.parse_args()
    print(args)
    main(args)