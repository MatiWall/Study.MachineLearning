import torch
from torch import nn

from .head import Head, MultiHeadAttention
from .feed_forward import FeedForward

class BlockEncoder(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_embeddings: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        
        
        encoder_heads = [Head(embedding_dim=embedding_dim, attention_dim=attention_dim, is_masked=False) for _ in range(4)]
        self.multihead_attention = MultiHeadAttention(
            heads=encoder_heads,
            embedding_dim=self.embedding_dim, 
            attention_dim=self.attention_dim
            )
        self.feedforward = FeedForward(num_embeddings=num_embeddings)
        
        self.layer_norm_attention = nn.LayerNorm(num_embeddings)
        self.layer_norm_feedforward = nn.LayerNorm(num_embeddings)
        
    def forward(self, x: torch.Tensor):
        x = self.layer_norm_attention(x + self.multihead_attention(x, x, x))
        x = self.layer_norm_feedforward(x + self.feedforward(x))
        return x
    
    
class BlockDecoder(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int, num_embeddings: int):
        super().__init__() 
        
        heads_masked = [Head(embedding_dim=embedding_dim, attention_dim=attention_dim, is_masked=True) for _ in range(4)]
        self.attention_masked = MultiHeadAttention(heads=heads_masked, attention_dim=attention_dim, embedding_dim=embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        heads_encoder = [Head(embedding_dim=embedding_dim, attention_dim=attention_dim, is_masked=True) for _ in range(4)]
        self.attention_encoder = MultiHeadAttention(heads=heads_encoder, attention_dim=attention_dim, embedding_dim=embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        
        self.ff = FeedForward(num_embeddings=num_embeddings)
        self.layer_norm_3 = nn.LayerNorm(embedding_dim)
        
    
    def forward(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor):
        
        decoder_input = self.layer_norm_1( decoder_input + self.attention_masked(decoder_input, decoder_input, decoder_input))
        
        decoder_input = self.layer_norm_2( decoder_input + self.attention_encoder(decoder_input, encoder_input, encoder_input) )
        
        decoder_input = self.layer_norm_3(decoder_input + self.ff(decoder_input))
        
        return decoder_input
        
        
class Decoder(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int, num_embeddings: int ):
        super().__init__()
        self.n_layers = 4
        
        self.layers = nn.ModuleList([
            BlockDecoder(embedding_dim=embedding_dim, attention_dim=attention_dim, num_embeddings=num_embeddings)
        ])
        
    def forward(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor):
        
        for layer in self.layers:
            decoder_input  = layer(decoder_input=decoder_input, encoder_input=encoder_input)
            
        return decoder_input
        

class Model(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int, num_embeddings: int, nx: int, block_size: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.nx = nx
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(num_embeddings, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        
        self.encoder = nn.Sequential(*[BlockEncoder(embedding_dim=embedding_dim, attention_dim=attention_dim, num_embeddings=num_embeddings) for _ in range(nx)])
        
        self.decoder = Decoder(embedding_dim=embedding_dim, attention_dim=attention_dim, num_embeddings=num_embeddings)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        
        x = token_embeddings + position_embedding
        # x ()
        encoder_output = self.encoder(x)
        
        decoder_output = self.decoder(decoder_input=x, encoder_output=encoder_output)
        
        
        
        return decoder_output
        
    
        

        
        
        
        
  