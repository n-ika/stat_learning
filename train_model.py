import torch; torch.manual_seed(108)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import pandas as pd
import librosa
import os
from ae_zevin import *
import sys

model_type = sys.argv[1]
exp = int(sys.argv[2])

if exp==1:
    root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_one/'
elif exp==2:
    root = '/Users/nika/Desktop/hippocampus/stat_learning/zevin/modeling/simulation_two/'
os.makedirs(root+'models',exist_ok=True)
os.makedirs(root+f'models/{model_type}',exist_ok=True)

train_data = torch.load(root+f'{model_type}_train_data.pt',weights_only=True)
sequential_syllables = torch.load(root+f'{model_type}_train_data_sequential.pt',weights_only=True)

epochs=8192
for NUM in range(0,50):
  loss_track = []

  model = AE(input_shape=train_data.shape[1],hidden_size=512,num_layers=7)
  loss_fun = torch.nn.MSELoss()
    # loss_fun = torch.nn.
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

  for epoch in range(1,epochs+1):
    # Forward
    codes, decoded = model(train_data)

    # Backward
    optimizer.zero_grad()
    loss = loss_fun(decoded, sequential_syllables)
    loss_track.append(float(loss))
    loss.backward()
    optimizer.step()

    # Show progress
    print('[{}/{}] Loss:'.format(epoch, epochs), loss.item())
    if epoch in np.logspace(0, 13, num=14, base=2, dtype=int):
        # Save
        torch.save(model.state_dict(), root+f'models/{model_type}/ckpt_'+str(epoch)+'_exp-'+str(exp)+'_'+str(NUM)+'.pt')
  pd.DataFrame(loss_track).to_csv(root+f'models/{model_type}/loss_exp-'+str(exp)+'_'+str(NUM)+'.csv')
  print('done model :',NUM)