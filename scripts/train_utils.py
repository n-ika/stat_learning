import torch
from torch.utils.data import  DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import re



def load_syll_vec(root_exp,input_type,model_type,stim_type):
    syll_vec = torch.load(root_exp+f'{input_type}/syll_vec_{model_type}_{stim_type}.pt',weights_only=False)
    return(syll_vec)

class CreateDataset():
    def __init__(self, data, labels, conditions=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = np.array(labels, dtype=str)
        if conditions is not None:
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


def extract_errors(in_stim,out_stim,lossf,device,model=None,hidden=None):
    model.eval()
    with torch.no_grad():
        in_stim = in_stim.to(torch.float32).to(device)
        out_stim = out_stim.to(torch.float32).to(device)
        if model.__class__.__name__ == 'AE':
            code, recon = model(in_stim)
        elif model.__class__.__name__ == 'RNNModel':
            h = hidden.clone() if hidden is not None else None
            recon, hidden = model(in_stim, h)
        total_loss = float(lossf(recon, out_stim).cpu().detach().numpy())
        recon = recon.cpu().detach().numpy()
        return(recon,total_loss)

def write_out_res(res,in_stim,out_stim,in_name,out_name,condition,lossf,device,model,hidden=None):
    recon, loss_item = extract_errors(in_stim, out_stim, lossf, device, model=model, hidden=hidden)
    res['loss'].append(loss_item)
    res['in_label'].append(in_name)
    res['out_label'].append(out_name)
    res['condition'].append(condition)
    return(res)

def test_model(test_in_loader,test_out_list,model,device,batch_id,lossf,epoch_num,stim_type,model_type,lr,exp,simulation_num,hidden=None):
    res = {'condition':[],'loss':[],'in_label':[],'out_label':[]}
    for i,test_in in enumerate(test_in_loader):
        in_batch = test_in[0].to(device)
        # in_batch = test_in[0].reshape(1,1,-1) #FIXME
        in_label = str(test_in[1][0])
        out_batch = test_out_list[i][0].to(device)
        out_label = str(test_out_list[i][1][0])
        if test_in[2].shape[0]>1:
            print('WARNING: more than one condition in test batch, taking first one only')
            condition = int(test_in[2][0])
        else:
            condition = int(test_in[2])
        res = write_out_res(res,
                            in_batch,
                            out_batch,
                            in_label,
                            out_label,
                            condition,
                            lossf,
                            device,
                            model,
                            hidden=hidden)
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
                test_frequency,
                stim_type,
                model_type,
                lr,
                exp,
                simulation_num,
                model_dir,
                out_dir,
                out_name
                ):
    loss_track = []

    # Check if any results already exist
    files = [f for f in os.listdir(f"{out_dir}") if f.startswith(out_name) and f.endswith(".csv")]
    if files and len(files) > 1:
        latest_res = max(files, key=lambda f: int(re.search(r"ep-(\d+)\.csv$", f).group(1))) \
                     if files else None
        latest_ep = int(re.search(r"ep-(\d+)\.csv$", latest_res).group(1))
        if latest_ep > 0 and latest_ep < epochs:
            print(f'Results for #{simulation_num} already exist. Starting from epoch {latest_ep}.')
            ckpt = torch.load(f'{model_dir}/{out_name}_ep-{latest_ep}.pt', map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif latest_ep == epochs:
            print(f'All epochs for #{simulation_num} already completed. Skipping training.')
            return()
        else:
            print("ERROR: Did not think about this case, TODO")
    # Else, start training from scratch
    else:
        print(f'Starting training for #{simulation_num} from scratch.')
        test_df = test_model(test_in_loader,test_out_list,model,device,0,loss_fun,
                            0,stim_type,model_type,lr,exp,simulation_num,None)  
        test_df.to_csv(out_dir+f'{out_name}_ep-0.csv',compression='gzip',index=False)
        latest_ep = 1
    # Start training loop
    for epoch in range(latest_ep,epochs+1):
        test_dfs = []
        hidden = None 
        for i, in_batch_ in enumerate(in_loader):
            model.train()
            out_batch = out_loader_list[i][0].to(device)
            in_batch = in_batch_[0].to(device)
            optimizer.zero_grad()

            avg_loss = 0.0
            for j in range(in_batch.shape[0]):
                if hidden is not None:
                    hidden = hidden.detach()
                decoded, hidden = model(in_batch[j:j+1, :], hidden)
                # decoded = decoded.unsqueeze(1) #TODO     
                assert decoded.shape == out_batch[j:j+1, :].shape, f"Shape mismatch: decoded {decoded.shape}, out_batch {out_batch[j:j+1, :].shape}"
                loss = loss_fun(decoded,out_batch[j:j+1, :])
                loss = loss / in_batch.shape[0]  # Average loss over the batch
                loss.backward()
                avg_loss += float(loss)

            loss_track.append(avg_loss)
            optimizer.step()
            if i%test_frequency==0: #FIXME 
                test_dfs.append(test_model(test_in_loader,test_out_list,model,device,i,loss_fun,
                                            epoch,stim_type,model_type,lr,exp,simulation_num,hidden=hidden))
        test_df_epoch = pd.concat(test_dfs)
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    }, 
                    os.path.join(model_dir,f'{out_name}_ep-{epoch}.pt'))             
        # torch.save(model.state_dict(), model_dir+f'{out_name}_ep-{epoch}.pt')
        test_df_epoch.to_csv(os.path.join(out_dir,f'{out_name}_ep-{epoch}.csv'),compression='gzip',index=False)
        pd.DataFrame(loss_track).to_csv(os.path.join(out_dir,f'loss_{out_name}.csv'),compression='gzip',index=False)
        print(f'done epoch & test #: {epoch} for #{simulation_num}')  
    return()