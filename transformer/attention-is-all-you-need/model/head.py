from torch import nn
import torch


class Head(nn.Module):
    """ Scaled dot product attention
    
    Implementation of the scaled dot product attention from the original paper
    
    """
    def __init__(self, embedding_dim: int, attention_dim: int, is_masked: bool = True ):
        super().__init__()
        self.is_masked = is_masked
        # Linear layers to transform the input embeddings into queries, keys and values. 
        # Projects the input embedding into a space of dim attention_dim
        # Does y = xA^T + b
        # W@x^T, but x is [num_embeddings, embedding_dim] -> W [embedding_dim, attention_dim]
        # attention_dim also know as d_k or head_dim, i.e. the size of the vectors used inside the attention mechanism.
        # Dimension is changed to learn better relationships.
        self.wq = nn.Linear(embedding_dim, attention_dim, bias=False) 
        self.wk = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, attention_dim, bias=False)
        
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, C = query.shape
        
        query = self.wq(query)
        keys = self.wk(key)
        values = self.wv(value)
        
        attention_score = query@keys.T /keys.shape[1]**0.5
        
        if self.is_masked: # To avoid forward looking
            upper_triangle = torch.triu(attention_score, diagonal=1).bool()
            attention_score[upper_triangle] = float("-inf") # Making sure softmax works
        
        attention_score_softmax = nn.functional.softmax(attention_score, dim=-1) # Column wise softmax
        
        weighted_values = attention_score_softmax @ values
        
        return weighted_values
    
    
class MultiHeadAttention(nn.Module):
    """
    Doing multiple attentions in parallel. Concat them and use a linear transformation to project to a lower dim.
    """
    def __init__(self, heads: list[Head], embedding_dim: int, attention_dim: int):
        super().__init__()
        self.heads = heads
        
        # projection to move from dimension of attention_dim to embedding dim
        self.projection = nn.Linear(in_features=attention_dim * len(heads), out_features=embedding_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor): 
        
        heads_evaluated = torch.concat([h(query=query, key=key, value=value) for h in self.heads], dim=-1)
        
        return self.projection(heads_evaluated) 
        

if __name__ == "__main__":
    num_embeddings = 7
    embedding_dim = 5
    attention_dim = embedding_dim * 2
    embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    

    
    random_vector = torch.randint(low=0, high=num_embeddings - 1, size=(num_embeddings,))
    random_embedding = embedding(random_vector)
    
    head = Head(embedding_dim=embedding_dim, attention_dim=attention_dim)
    res = head(random_embedding)
    
        
    heads = [Head(embedding_dim=embedding_dim, attention_dim=attention_dim) for _ in range(4)]
    multihead_attention = MultiHeadAttention(heads=heads, embedding_dim=embedding_dim, attention_dim=attention_dim)
    
    res = multihead_attention(random_embedding)
    
    pass