import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_causal_mask(seq_len: int, device=None):
    """
    return mask of shape [1,1,seq_len, seq_len]
    True = allowed, False = blacked
    """
    m = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return m.view(1, 1, seq_len, seq_len)

# Reference 
# torch.ones(seq_len, seq_len) makes a square matrix of 1s.
# torch.tril(...) keeps the lower triangle (including diagonal).
# We store it as bool: True where attention is allowed.

def make_padding_mask(attn_mask_1d: torch.Tensor):
    """ 
    attn_mask_1d: [B, T] where True = keep token, False = pad token
    Returns [B, 1, 1, T] broadcastable to attention logits
    """
    return attn_mask_1d[:, None, None, :].to(torch.bool)

# insets 2 singleton dims, and sends it back to [B, H, Tq, Tk]
# for each batch, which key positions are valid

class ScaledDotProductAttention(nn.Module):
    """
    Q: [B, H, Tq, Dh]
    K: [B, H, Tk, Dh]
    V: [B, H, Tk, Dh] 
    mask (optional) should be broadcastable to [B, H, Tq, Tk]
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        Dh = Q.size(-1)
        logits = (Q @ K.transpose(-2,-1)) / math.sqrt(Dh) # [B, H, Tq, Tk]
        if attn_mask is not None:
            # attn_mask: True = keep, False = block
            logits = logits.masked_fill(~attn_mask, float("-inf"))

        weights = F.softmax(logits, dim=-1) # [B, H, Tq, Tk]
        weights = self.dropout(weights)
        out = weights @ V # [B, H, Tq, Dh]
        return out, weights
    