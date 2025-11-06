import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np

def create_encoder(input_size, hidden_size, bottleneck_size, num_layers):
    layers=[]
    current_size = hidden_size
    input_shape = np.prod(input_size)
    layers.append(nn.Flatten())
    layers.append(nn.Linear(input_shape,hidden_size))
    layers.append(nn.ReLU())
    while current_size // 2 >= bottleneck_size:
        if len(layers)//2 < num_layers - 1:
            # if current_size > bottleneck_size:
            next_size = current_size // 2
            layers.extend([
                nn.Linear(current_size, next_size),
                nn.ReLU()
            ])
            current_size = next_size
        else:
            layers.extend([
                nn.Linear(current_size, bottleneck_size),
                nn.ReLU()
            ])
            current_size = bottleneck_size
    remaining_layers = num_layers - (len(layers)//2)
    for i in range(remaining_layers):
        layers.extend([
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.ReLU()
        ])
    return layers


def create_decoder(encoder_layers, output_size, sigmoid):
    decoder_linears = [nn.Linear(layer.out_features, layer.in_features) 
                       if isinstance(layer, nn.Linear) else nn.ReLU()
                       for layer in reversed(encoder_layers[2:])
                       ]
    # del decoder_linears[-1]
    decoder_linears.append(nn.Linear(encoder_layers[1].out_features, np.prod(output_size)))
    decoder_linears.append(nn.Unflatten(dim=1, unflattened_size=tuple(output_size)))
    if sigmoid==True:
        decoder_linears.append(nn.Sigmoid())
    return(decoder_linears[1:])


class AE(torch.nn.Module):
    
    def __init__(self,input_size, output_size, hidden_size=512, num_layers=7, bottleneck_size=8, sigmoid=True):
        super(AE, self).__init__()
        encoder_layers = create_encoder(input_size=input_size, 
                                        hidden_size=hidden_size,
                                        bottleneck_size=bottleneck_size, 
                                        num_layers=num_layers)
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*create_decoder(encoder_layers,
                                        output_size=output_size, 
                                        sigmoid=sigmoid))

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return(encoded,decoded)



class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=8, loss_type='bce'):
        super().__init__()
        self.loss_type = loss_type
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, h0=None):
        if h0 != None:
            out, _ = self.rnn(x,h0)
        else:
            out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        if self.loss_type == 'bce':
            out = self.sigmoid(out)
        elif self.loss_type == 'mse':
            out = torch.relu(out)
        else:
            raise ValueError("Unsupported loss type. Use 'bce' or 'mse'.")
        return out
    