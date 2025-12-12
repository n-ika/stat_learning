import torch
from torch.utils.data import  DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os


def load_syll_vec(root_exp,input_type,model_type,stim_type):
    syll_vec = torch.load(root_exp+f'{input_type}/syll_vec_{model_type}_{stim_type}.pt',weights_only=False)
    # syll_vec = reshape_syll_vec(syll_vec_)
    return(syll_vec)

# def reshape_syll_vec(syll_vec_):
#     syll_vec = {}
#     for key in syll_vec_.keys():
#         syll_vec[key] = torch.tensor(syll_vec_[key].astype(np.float32).reshape(1, 1, -1))
#     return(syll_vec)


class CreateDataset():
    def __init__(self, data, labels, conditions=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = np.array(labels, dtype=str)
        if conditions != None:
            self.conditions = torch.tensor(conditions, dtype=torch.long)
            assert len(self.data) == len(self.conditions)
        else:
            self.conditions = None
        assert len(self.data) == len(self.labels)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.conditions is None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx], self.conditions[idx]


def load_data(name,data_path,batch_size):
    data_raw = torch.load(data_path+f'_{name}.pt',weights_only=False)
    if 'conditions' in data_raw.keys():
        dataset_loaded = CreateDataset(data_raw['inputs'], data_raw['labels'], data_raw['conditions'])
    else:
        dataset_loaded = CreateDataset(data_raw['inputs'], data_raw['labels'])
    data_loader = DataLoader(dataset_loaded, batch_size=batch_size, shuffle=False, pin_memory=True)
    return(data_loader)

def load_all_data(root_exp,model_type,input_type,batch_size_train,batch_size_test):
    data_path = root_exp+f'{input_type}/{model_type}'
    syll_vec = load_syll_vec(root_exp, input_type, model_type,stim_type=input_type.split('_')[0])
    in_loader = load_data('train_in',
                            data_path,
                            batch_size_train)
    out_loader = load_data('train_out',
                            data_path,
                            batch_size_train)
    test_in_loader = load_data('test_in',
                                data_path,
                                batch_size_test)
    test_out_loader = load_data('test_out',
                                    data_path,
                                    batch_size_test)
    return(syll_vec,in_loader,out_loader,test_in_loader,test_out_loader)


def extract_errors(in_stim,out_stim,lossf,device,model=None):
    model.eval()
    in_stim = in_stim.to(torch.float32).to(device)
    out_stim = out_stim.to(torch.float32).to(device)
    if model.__class__.__name__ == 'AE':
        code, recon = model(in_stim)
    elif model.__class__.__name__ == 'RNNModel':
        h0 = torch.zeros(in_stim.shape[0], 1, model.hidden_size).to(device)
        recon = model(in_stim, h0)
        recon = recon.unsqueeze(1)
    total_loss = float(lossf(recon, out_stim).cpu().detach().numpy())
    recon = recon.cpu().detach().numpy()
    return(recon,total_loss)

def write_out_res(res,in_stim,out_stim,in_name,out_name,condition,lossf,device,model=None):
    recon, loss_item = extract_errors(in_stim, out_stim, lossf, device, model=model)
    # print('recon', recon)
    res['loss'].append(loss_item)
    res['in_label'].append(in_name)
    res['out_label'].append(out_name)
    res['condition'].append(condition)
    return(res)

def test_model(test_in_loader,test_out_list,model,device,batch_id,lossf,epoch_num,stim_type,model_type,lr,exp,simulation_num):
    res = {'condition':[],'loss':[],'in_label':[],'out_label':[]}
    for i,test_in in enumerate(test_in_loader):
        in_batch = test_in[0]
        # in_batch = test_in[0].reshape(1,1,-1) #FIXME
        in_label = str(test_in[1][0])
        out_batch = test_out_list[i][0]
        out_label = str(test_out_list[i][1][0])
        condition = int(test_in[2])
        res = write_out_res(res,
                            in_batch,
                            out_batch,
                            in_label,
                            out_label,
                            condition,
                            lossf,
                            device,
                            model)
    df=pd.DataFrame(res)
    df['batch_id']=batch_id
    df['epochs']=epoch_num
    df['stim_type'] = stim_type
    df['model_type'] = model_type
    df['lr'] = lr
    df['exp'] = exp
    df['model_num'] = simulation_num
    return(df)

def train_model(model,
                optimizer,
                loss_fun,
                device,epochs,
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
                out_dir,
                out_name):
    loss_track = []
    test_df = test_model(test_in_loader,test_out_list,model,device,0,loss_fun,
                            0,stim_type,model_type,lr,exp,simulation_num)  
    test_df.to_csv(out_dir+f'{out_name}_ep-0.csv')
    if os.path.exists(f'{out_dir}/{out_name}_ep-{epochs}.csv'):
        # Results already exist; skip training
        print(f'Results for {out_name} already exist. Skipping training.')
        return()        
    else:
        for epoch in range(1,epochs+1):
            test_dfs = []
            for i, in_batch_ in enumerate(in_loader):
                model.train()
                out_batch = out_loader_list[i][0].to(device)
                # in_batch = in_batch_[0].reshape(1,1,-1).to(device) #FIXME
                in_batch = in_batch_[0].to(device)
                if model.__class__.__name__ == 'AE':
                    codes, decoded = model(in_batch)
                elif model.__class__.__name__ == 'RNNModel':
                    # h0 = torch.zeros(in_batch.shape).to(device)
                    decoded = model(in_batch)
                    decoded = decoded.unsqueeze(1)
                optimizer.zero_grad()
                loss = loss_fun(decoded,out_batch)
                loss_track.append(float(loss))
                loss.backward()
                optimizer.step()
                # if batch_size_test==1: #FIXME uncomment
                #     test_dfs.append(test_model(test_in_loader,test_out_list,model,device,i,loss_fun,
                #                                epoch,stim_type,model_type,lr,exp,simulation_num))
                # if i > 2: #FIXME - remove
                #     break
            # if batch_size_test>1: #FIXME uncomment
            test_dfs.append(test_model(test_in_loader,test_out_list,model,device,0,loss_fun,
                                           epoch,stim_type,model_type,lr,exp,simulation_num))
            test_df_epoch = pd.concat(test_dfs)             
            torch.save(model.state_dict(), model_dir+f'{out_name}_ep-{epoch}.pt')
            test_df_epoch.to_csv(out_dir+f'{out_name}_ep-{epoch}.csv',compression='gzip',index=False)
            pd.DataFrame(loss_track).to_csv(out_dir+f'loss_{out_name}.csv',compression='gzip',index=False)
            print('done epoch & test #: ',epoch)  
    return()