import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, context_len):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)
        # Register buffer for causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(context_len, context_len) * float("-inf"), diagonal=1
            ),
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        norm_x = self.ln1(x)
        current_mask = self.causal_mask[:seq_len, :seq_len]
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x, attn_mask=current_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x


class TextGenModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_layers=4, num_heads=4):
        super(TextGenModel, self).__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, context_len)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight  # weight tying.

        self.register_buffer("pos_ids", torch.arange(context_len, dtype=torch.long))

    def forward(self, x):
        batch, seq_len = x.shape
        token_embeds = self.embedding(x)
        positions = self.pos_embedding(self.pos_ids[:seq_len])
        x = token_embeds + positions
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.fc_out(x)
        return logits
