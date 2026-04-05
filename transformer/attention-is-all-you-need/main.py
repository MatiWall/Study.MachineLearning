from dataclasses import dataclass
import torch
from torch import nn 

@dataclass
class Config:
    nr_embeddings: int
    embedding_dim: int
    
    
    
    
def main():
    
    config = Config(
        nr_embeddings=10,
        embedding_dim=5
    )
    
    
    print(config)
    
    
if __name__ == "__main__":
    main()