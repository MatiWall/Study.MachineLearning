import torch
import torch.nn as nn



class BigramLanguageModel(nn.Module):
    def __init__(self, nr_tokens: int):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(nr_tokens, nr_tokens)
        
    def forward(self, idx ,targets=None):
        
        logits = self.token_embedding_table(idx)
    
        
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
            logits, loss = self(idx)
            
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
    
    # Test encoding and decoding
    test_string = "Hello there"
    encoded = encode(test_string)
    print(f"Encoded '{test_string}': {encoded}")
    print(f"Decoded {encoded}: {decode(encoded)}")
    
    data = torch.tensor(encode(data), dtype=torch.long)
    
    fraction = int(0.9 * len(data))
    train_data = data[:fraction]
    val_data = data[fraction:]
    
    block_size = 8
    nr_batches = 4
    
    
    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    for i in range(block_size):
        context = x[:i+1]
        target = y[i]
        print(f"Context: {context}, Target: {target}")
    

    batches = get_batch(train_data, nr_batches=nr_batches, block_size=block_size)
    
    print(batches)
    
    model = BigramLanguageModel(nr_tokens=len(tokens))
    
    logits, loss = model(batches[0], batches[1])
    
    print(loss)
    
    idx = torch.zeros((1,1), dtype=torch.long)
    result = model.generate(idx, max_new_tokens=100)
    
    print(decode(result[0].tolist()))

    # Train    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    steps = 100
    nr_batches = 40
    for step in range(steps):
        xb, yb = get_batch(train_data, nr_batches=nr_batches, block_size=block_size)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()
        print(f"step {step}, loss {loss.item()}")
    pass
    
if __name__ == "__main__":
    main()