import torch
import torch.nn as nn


block_size = 128
batch_size = 32
steps = 5000
nr_batches = 40
n_embeddings = 32
learning_rate = 3e-4
n_blocks = 4
dropout = 0.2


torch.manual_seed(42)


class Head(nn.Module):
    def __init__(self, n_embeddings: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        k = self.key(x) # B, T, C
        q = self.query(x)
        
        weight = q  @ k.transpose(-2, -1) * C**-0.5
        
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        weight = nn.functional.softmax(weight, dim=-1)
        
        x = self.dropout(x)
        v = self.value(x)
        return weight @ v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int, block_size: int, n_embeddings: int):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(n_embeddings, head_size, block_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        
        x = torch.concat([h(x) for h in self.heads], dim=-1)
        x = self.projection(x)
        return self.dropout(x)
    
class FeedForward(nn.Module):
    
    def __init__(self, n_embeddings: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings), 
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
    
class Block(nn.Module):
    
    def __init__(self, n_embeddings: int, n_heads, head_size: int):
        super().__init__()
        
        self.sa_head = MultiHeadAttention(n_heads=n_heads, head_size=head_size, block_size=block_size, n_embeddings=n_embeddings)
        self.ff = FeedForward(n_embeddings=n_embeddings)
        
        self.layer_norm_1 = nn.LayerNorm(n_embeddings)
        self.layer_norm_2 = nn.LayerNorm(n_embeddings)
        
    def forward(self, x: torch.Tensor):
        
        x = x + self.sa_head(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        
        return x
        
        
        

class BigramLanguageModel(nn.Module):
    def __init__(
        self, 
        nr_tokens: int, 
        n_embeddings: int, 
        block_size: int,
        n_heads: int,
        head_size: int,
        n_blocks: int
        ):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(nr_tokens, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, nr_tokens)
        self.blocks = nn.Sequential(
            *[Block(n_embeddings=n_embeddings, n_heads=n_heads, head_size=head_size) for _ in range(n_blocks)],

            nn.LayerNorm(n_embeddings)
        )
        
    def forward(self, idx ,targets=None):
        
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx) # B, T, C
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embeddings + position_embedding
        x = self.blocks(x)
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
        

def get_batch(data: torch.Tensor, nr_batches: int, block_size: int, batch_size: int):
    batch_idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    
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
    
    model = BigramLanguageModel(
        nr_tokens=len(tokens), 
        n_embeddings=n_embeddings, 
        block_size=block_size,
        head_size=4,
        n_heads=8, 
        n_blocks=n_blocks
        )
    

    # Train    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for step in range(steps):
        xb, yb = get_batch(train_data, nr_batches=nr_batches, block_size=block_size, batch_size=batch_size)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()
        print(f"step {step}, loss {loss.item()}")

    idx = torch.zeros((1,1), dtype=torch.long)
    result = model.generate(idx, max_new_tokens=1000)

    print(decode(result[0].tolist()))
    
if __name__ == "__main__":
    main()