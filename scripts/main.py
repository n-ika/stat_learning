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
    unique_init = args.unique_init
    simulation_num = args.simulation_num
    optional = args.optional
    encoding_types = args.encoding_types
    stim_structure = args.stim_structure
    # all_dfs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

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

    for model_type in encoding_types: 
        for stim_type in stim_structure: 
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
                syll_vec,in_loader,out_loader,test_in_loader,test_out_loader = load_all_data(root_exp,
                                                                                             model_type,
                                                                                             input_type,
                                                                                             batch_size_train,
                                                                                             batch_size_test)
                test_out_list = list(test_out_loader)
                out_loader_list = list(out_loader)
                batch_sample = next(iter(in_loader))
                
                if model_arch == 'AE':
                    if model_type.startswith('acoustic_vec'):
                        out_size = [1,int(model_type.split('_')[-1])] #TODO
                    else:
                        out_size = syll_vec['pi'].shape[1:]
                    model = AE(input_size=batch_sample[0].shape[1:],
                                output_size=out_size,
                                hidden_size=hidden_size,
                                num_layers=num_layer,
                                bottleneck_size=bottleneck_size,
                                sigmoid=sigmoid).to(device)
                elif model_arch == 'RNN':
                    if unique_init == True:
                        torch.manual_seed(842)  # Ensure the same initialization for each model run
                    if model_type.startswith('acoustic_vec'):
                        out_size = int(model_type.split('_')[-1])
                    else:
                        out_size = np.prod(syll_vec['pi'].shape[1:])
                    model = RNNModel(input_size=batch_sample[0].shape[-1],
                                        output_size=out_size,
                                        hidden_size=hidden_size,                                         
                                        loss_type=loss_type).to(device)
                optimizer = torch.optim.Adam(model.parameters(),lr=lr)
                print(f'Creating model: {model_type}, lr: {lr}, experiment: {exp}, stimulus type: {stim_type},'
                        f'number: {simulation_num}, train batch: {batch_size_train}, test batch: {batch_size_test}')   
                out_name = (
                        f"{model_type}_{bottleneck_size}_exp-{exp}_n-{simulation_num}_lr-{lr}_train-{int(batch_size_train)}_test"
                        f"-{int(batch_size_test)}_{optimizer.__class__.__name__}"
                        f"_{loss_type}_{sigmoid_type}"
                    )
                # If using unique initialization, load or save initial model state
                if unique_init==True:
                    if os.path.exists(root_dir+f'/init_{loss_type}_{stim_type}_{model_type}.pt'):
                        print('LOADING initial model state...')
                        ckpt_init = torch.load(root_dir+f'/init_{loss_type}_{stim_type}_{model_type}.pt', 
                                               map_location=device,
                                               weights_only=False)
                        model.load_state_dict(ckpt_init["model_state_dict"])
                        optimizer.load_state_dict(ckpt_init["optimizer_state_dict"])
                    else:
                        print('SAVING initial model state...')
                        torch.save({"epoch": 0,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": 0,
                        }, 
                        root_dir+f'/init_{loss_type}_{stim_type}_{model_type}.pt')    
                else:
                    print('New initialization ...') 

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
                            out_name,
                            unique_init)
                # all_dfs.append(test_df)
                print(f'done model #: {simulation_num},experiment: {exp}, stimulus type: {stim_type}, model type: {model_type}')
    # df_all = pd.concat(all_dfs)
    # df_all.to_csv(root_dir+f'/all_results_{model_arch}_{loss_type}_{sigmoid_type}.csv')
    print('done all!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', '-ir', type=str, help='Root directory for data', 
                        default='/projects/jurovlab/stat_learning/data/')
    parser.add_argument('--out_root', '-or', type=str, help='Root directory for results', 
                        default='/projects/jurovlab/stat_learning/interim/')
    parser.add_argument('--model_arch', '-ma', type=str, choices=['AE', 'RNN'], default='AE', help='Model architectures to run')
    parser.add_argument('--batch_size_train', '-btr', type=int, default=1, help='Training batch size')
    parser.add_argument('--batch_size_test', '-bte', type=int, default=1, help='Testing batch size')
    parser.add_argument('--bottleneck_size', '-bs', type=int, default=8, help='Size of the bottleneck layer')
    parser.add_argument('--hidden_size', '-hs', type=int, default=512, help='Size of the hidden layers')
    parser.add_argument('--num_layer', '-nl', type=int, default=7, help='Number of layers in the model')
    parser.add_argument('--epochs', '-e', type=int, default=16, help='Number of training epochs')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--loss_type', '-lt', type=str, choices=['mse', 'bce'], default='mse', help='Type of loss function')
    parser.add_argument('--unique_init', '-ui', action="store_true", help='Use the same initialization for each model run')
    parser.add_argument('--simulation_num', '-sn', type=int, default=0, help='Number of simulation run')
    parser.add_argument('--optional', '-o', type=str, default='', 
                        help='Optional out folder descriptor (output folder is [model type]_[loss type][optional])')
    parser.add_argument('--encoding_types', '-et', nargs='+', default=['onehot', 'phon', 'acoustic_39_norm'], 
                        help='List of encoding types to use')
    parser.add_argument('--stim_structure', '-st', nargs='+', default=['unigram', 'zerovec-bigram', 'bigram'], 
                        help='List of stimulus structure to use')
    args = parser.parse_args()
    print(args)
    main(args)