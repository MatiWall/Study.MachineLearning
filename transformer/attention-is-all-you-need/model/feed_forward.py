from torch import nn
import torch

class FeedForward(nn.Module):
    
    def __init__(self, num_embeddings: int):
        super().__init__()
        
        self.ff = nn.Sequential(
            nn.Linear(in_features=num_embeddings, out_features=num_embeddings * 4), # multiple of 4 from "Attention is all you need"
            nn.ReLU(),
            nn.Linear(in_features=num_embeddings * 4, out_features=num_embeddings),
        )
        
        
    def forward(self, x: torch.Tensor):
        return self.ff(x)