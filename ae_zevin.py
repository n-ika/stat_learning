import torch; torch.manual_seed(108)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

def create_encoder(input_size, hidden_size, num_layers):
    layers = []
    current_size = hidden_size
    layers.append(nn.Linear(input_size,hidden_size))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 1):
        next_size = current_size // 2
        layers.extend([
            nn.Linear(current_size, next_size),
            nn.ReLU()
        ])
        current_size = next_size
    return nn.Sequential(*layers)

def create_decoder(input_size, output_size, num_layers):
    layers = []
    current_size = input_size
    for _ in range(num_layers - 1):
        next_size = current_size * 2
        layers.extend([
            nn.Linear(current_size, next_size),
            nn.ReLU()
        ])
        current_size = next_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)

class AE(torch.nn.Module):
    
    def __init__(self,input_shape,hidden_size=1024, num_layers=7):
        super(AE, self).__init__()
        self.encoder = create_encoder(input_size=input_shape, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = create_decoder(input_size=list(self.encoder.parameters())[-1].shape[0], output_size=input_shape, num_layers=num_layers)

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return(encoded,decoded)
    

class AE_1hot(torch.nn.Module):
    def __init__(self):
        super(AE_1hot, self).__init__()

        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(12,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,32), # remplace par SITH layer
                torch.nn.ReLU())
        
        
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(32,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,12),
                torch.nn.Softmax()
                )
                # torch.nn.ReLU()
        
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return(encoded,decoded)