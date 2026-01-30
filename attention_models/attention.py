import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
B: Batch size, T: number of tokens, D: model dimension(embedding size), H: number attention heads
Dh: Dimension per head, Tq: Query seq len, Tk: key seq len, Q,K,V : tensor
"""

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

class MultiHeadSelfAttention(nn.Module):
    """ 
    Input: [B, T, D]
    Output: [B, T, D]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x):
        # x: [B, T, D] -> [B, H, T, Dh]
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1,2)
    
    def _merge_heads(self, x):
        # x: [B, H, T, dh] -> [B, T, D]
        B, H, T, Dh = x.shape
        x = x.transpose(1,2).contiguous().view(B, T, H * Dh)
        return x
    
    def forward(self, x, padding_mask_1d=None, causal=False, return_weights=False):
        """
        padding_mask_1d: [B, T] True=keep token, False= pad token
        causal: if True, prevent attending to future tokens
        """

        B, T, _ = x.shape
        device = x.device

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        Q = self._split_heads(q) # [B, H, T, Dh]
        K = self._split_heads(k)
        V = self._split_heads(v)

        attn_mask = None

        if padding_mask_1d is not None:
            pm = make_padding_mask(padding_mask_1d)  # [B, 1, 1, T]
            attn_mask = pm if attn_mask is None else (attn_mask & pm)
        
        if causal:
            cm = make_causal_mask(T, device=device) # [ 1, 1, T, T]
            attn_mask = cm if attn_mask is None else (attn_mask & cm)

        out, weights = self.attn(Q, K, V, attn_mask=attn_mask)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_weights:
            return out, weights
        return out


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 5, 32
    H = 4

    x = torch.randn(B, T, D)

    # Example padding mask: last 2 tokens of batch 0 are padding

    padding_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])    

    mha = MultiHeadSelfAttention(d_model=D, n_heads=H, dropout=0.1)
    y, w = mha(x, padding_mask_1d=padding_mask, causal=True, return_weights=True)

    print("y:", y.shape) # [B, T, D]
    print("w:", w.shape) # [B, H, T, T]
    print("weights row sums (should be ~1 where not all masked):", w[1,0,0].sum().item())
