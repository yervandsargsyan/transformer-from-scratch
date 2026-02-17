import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,  
        num_heads: int = 4,        
        num_layers: int = 4,       
        ff_hidden_dim: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, embedding_dim))  # max seq_len 1000

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

        # Output projection
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        returns: logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()

        # Embed tokens and add positional embeddings
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        # Transformer expects shape [seq_len, batch_size, embedding_dim]
        x = x.transpose(0, 1)

        x = self.transformer(x)

        # Back to [batch_size, seq_len, embedding_dim]
        x = x.transpose(0, 1)

        logits = self.fc_out(x)
        return logits
