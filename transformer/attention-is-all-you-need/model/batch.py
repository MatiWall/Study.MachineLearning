import torch

def get_batch(data: torch.Tensor, nr_batches: int, block_size: int, batch_size: int):
    batch_idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    
    x = torch.stack([data[b_idx:(b_idx + block_size)] for b_idx in batch_idx])
    y = torch.stack([data[(b_idx + 1):(b_idx + block_size + 1)] for b_idx in batch_idx])
    return x, y