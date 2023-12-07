import torch
import torch.nn as nn

# Define the Noise-to-W Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, final_dim, num_layers):
        super(MappingNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, final_dim))
        
        self.mapping_layers = nn.Sequential(*layers)
        
    def forward(self, noise):
        latent = self.mapping_layers(noise)
        return latent
