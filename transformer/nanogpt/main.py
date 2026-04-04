import torch
import torch.nn as nn


block_size = 8
n_embed = 32
steps = 5000
nr_batches = 40
n_embeddings = 32
learning_rate = 1e-3


class Head(nn.Module):
    def __init__(self, n_embeddings: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        k = self.key(x) # B, T, C
        q = self.query(x)
        
        weight = q  @ k.transpose(-2, -1) * C**-0.5
        
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        weight = nn.functional.softmax(weight, dim=-1)
        
        
        v = self.value(x)
        return weight @ v

class BigramLanguageModel(nn.Module):
    def __init__(self, nr_tokens: int, n_embeddings: int, block_size: int):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(nr_tokens, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, nr_tokens)
        self.sa_head = Head(n_embeddings, n_embeddings, block_size)
        
    def forward(self, idx ,targets=None):
        
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx) # B, T, C
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embeddings + position_embedding
        x = self.sa_head(x)
        logits = self.lm_head(x)
    
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, target=targets)
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            
            idx_crop = idx[:, -block_size:]
            
            logits, loss = self(idx_crop)
            
            logits = logits[:, -1, :]
            
            probabilities = nn.functional.softmax(logits, dim=1)
            
            idx_next = torch.multinomial(probabilities, num_samples=1)
            
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
        

def get_batch(data: torch.Tensor, nr_batches: int, block_size: int):
    batch_idx = torch.randint(0, len(data) - block_size - 1, (nr_batches,))
    
    x = torch.stack([data[b_idx:(b_idx + block_size)] for b_idx in batch_idx])
    y = torch.stack([data[(b_idx + 1):(b_idx + block_size + 1)] for b_idx in batch_idx])
    return x, y


def main():
    with open("shakespears.txt") as f:
        data = f.read()
    
    tokens = sorted(list(set(data)))
    
    print(f"Number of unique characters: {len(tokens)}")
    print(f"Characters: {tokens}")
    
    index_to_token = {i: t for i, t in enumerate(tokens)}
    token_to_index = {t: i for i, t in enumerate(tokens)}
    
    encode = lambda string: [token_to_index[x] for x in string]
    decode = lambda x: "".join([index_to_token[i] for i in x])
    

    
    data = torch.tensor(encode(data), dtype=torch.long)
    
    fraction = int(0.9 * len(data))
    train_data = data[:fraction]
    val_data = data[fraction:]
    
    model = BigramLanguageModel(nr_tokens=len(tokens), n_embeddings=n_embeddings, block_size=block_size)
    
    
    idx = torch.zeros((1,1), dtype=torch.long)
    result = model.generate(idx, max_new_tokens=100)
    
    print(decode(result[0].tolist()))

    # Train    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for step in range(steps):
        xb, yb = get_batch(train_data, nr_batches=nr_batches, block_size=block_size)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()
        print(f"step {step}, loss {loss.item()}")

    idx = torch.zeros((1,1), dtype=torch.long)
    result = model.generate(idx, max_new_tokens=100)

    print(decode(result[0].tolist()))
    
if __name__ == "__main__":
    main()