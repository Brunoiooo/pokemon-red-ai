import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=src_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


class TransformerMemory(nn.Module):
    """
    Episodic memory using Transformer for long-horizon planning.

    Processes sequences of game states to learn:
    - Location memory (seen places before)
    - Trainer strategies
    - Long-term dependencies (up to 2000 steps)
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ff_dim: int = 512,
        max_seq_len: int = 2000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(state_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_sequence: (batch_size, seq_len, state_dim) - sequence of game states

        Returns:
            context: (batch_size, d_model) - context-aware representation of the last state
        """
        x = self.input_projection(state_sequence)
        x = self.pos_encoding(x)

        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask)

        last_output = x[:, -1, :]
        return self.output_norm(last_output)

    def get_context_window(self, max_steps: int = 500) -> int:
        """
        Returns the effective context window in steps.
        Transformer can attend to full sequence up to max_seq_len.
        """
        return min(max_steps, self.max_seq_len)
