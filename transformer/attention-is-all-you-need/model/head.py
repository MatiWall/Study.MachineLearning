from torch import nn


class Head(nn.Model):
    
    def __init__(self, n_embeddings: int):
        super().__init__()
        
        self.query = nn.Linear(n_embeddings)
        self.keys = nn.Linear(n_embeddings)