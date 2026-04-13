from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

import pandas as pd

from model.batch import get_batch
from model.model import Model 

    
def main():
    
    with Path("./data/einstein.txt").open("r", encoding="utf-8") as f:
        text = f.read()
        
    tokens = sorted(list(set(text)))
    
    print(f"Number of unique characters: {len(tokens)}")
    print(f"Characters: {tokens}")
    index_to_token = {i: t for i, t in enumerate(tokens)}
    token_to_index = {t: i for i, t in enumerate(tokens)}
    
    encode = lambda string: [token_to_index[x] for x in string]
    decode = lambda x: "".join([index_to_token[i] for i in x])
    
    embedding_dim = 32
    attention_dim = 4
    num_embeddings=len(tokens)
    nx = 4
    
    block_size = 8
    batch_size = 32
    
    steps = 100
    
    
    model = Model(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        attention_dim=attention_dim,
        block_size=block_size,
        nx=nx
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.tensor(encode(text), dtype=torch.long)
    fraction = int(0.9 * len(data))
    train_data = data[:fraction]
    val_data = data[fraction:]
    
    for step in range(steps):
        xb, yb = get_batch(train_data, nr_batches=2, block_size=block_size, batch_size=batch_size)
        # xb/yb (batch_size, block_size)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        
        optimizer.step()
        print(f"step {step}, loss {loss.item()}")
    
if __name__ == "__main__":
    main()