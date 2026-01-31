import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    """Classic scaled dot-product attention used in Transformers.

    Every query softly weighs all keys/values to build a context vector, so the
    module excels when global information matters (e.g., encoder-decoder models).
    Masks can zero out invalid positions such as padding tokens or future tokens
    in autoregressive decoders.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class HardAttention(nn.Module):
    """Stochastic attention that approximates discrete alignments with Gumbel noise.

    Instead of softly mixing every position, this variant injects Gumbel noise and
    temperature scaling so each query focuses on a small subset of keys, which can
    better model sharp alignments (e.g., monotonic aligners) while remaining
    differentiable through the softmax.
    """

    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
        noisy_scores = (scores + gumbel_noise) / self.temperature
        attn_weights = F.softmax(noisy_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class SelfAttention(nn.Module):
    """Single-sequence attention that captures dependencies within a token stream.

    Input tokens are projected into Q/K/V spaces and each position attends to all
    other positions, which is the core building block of Transformer encoder/decoder
    stacks. Optional masks enable causal self-attention or padding suppression.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class CrossAttention(nn.Module):
    """Query sequence attending over an external context sequence.

    Commonly used in encoder-decoder setups where decoder queries pull information
    from encoder-produced keys/values. Masks restrict which context tokens are
    visible (e.g., padding or structured masks for retrieval).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention used in almost all modern Transformers.

    The input is split into `num_heads` subspaces so each head can learn distinct
    relations (syntax, long-range cues, etc.), then merged via an output projection.
    Supports both self- and cross-attention scenarios depending on the supplied
    query/key/value tensors and masks.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size = query.size(0)
        Q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = (
            torch.matmul(attn_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(output)
        return output, attn_weights


class AdditiveAttention(nn.Module):
    """Bahdanau (additive) attention widely used in early seq2seq models.

    Queries and keys are passed through learned linear layers, summed, then scored
    with a small neural network (`tanh` + vector projection). This formulation can
    be more expressive for small `d_model` values and works well in RNN-based
    encoder-decoder architectures.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        Q = self.W_q(query).unsqueeze(2)
        K = self.W_k(key).unsqueeze(1)
        scores = torch.tanh(Q + K)
        scores = self.v(scores).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiplicativeAttention(nn.Module):
    """Luong/general dot-product attention.

    Applies a learned linear map to keys before computing scaled dot products,
    enabling bilinear interactions between queries and keys. Frequently used in
    RNN encoder-decoder models where queries and keys live in the same space but
    benefit from an extra projection.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        scores = torch.matmul(query, self.W(key).transpose(-2, -1)) / (
            self.d_model**0.5
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class LocalAttention(nn.Module):
    """Sliding-window attention for long sequences where locality dominates.

    Each token only considers neighbors inside `window_size`, dramatically reducing
    compute and enforcing inductive bias toward local patterns (audio, time-series,
    long texts). Masks can further prune tokens inside the window.
    """

    def __init__(self, d_model: int, window_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = query.shape
        output = torch.zeros_like(query)
        attn_weights_all = torch.zeros(batch_size, seq_len, seq_len)

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)

            q_i = query[:, i : i + 1, :]
            k_local = key[:, start:end, :]
            v_local = value[:, start:end, :]

            scores = torch.matmul(q_i, k_local.transpose(-2, -1)) / (self.d_model**0.5)
            local_mask = mask[:, i : i + 1, start:end] if mask is not None else None
            if local_mask is not None:
                scores = scores.masked_fill(local_mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights_all[:, i, start:end] = attn_weights.squeeze(1)
            output[:, i : i + 1, :] = torch.matmul(attn_weights, v_local)

        return output, attn_weights_all


class FlashAttention(nn.Module):
    """Wrapper around PyTorch's fused scaled_dot_product_attention (FlashAttention style).

    Normalizes Q/K before invoking the kernel to improve numerical stability while
    obtaining reduced memory usage and better throughput on GPUs. Ideal when working
    with long sequences that still require global attention.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        Q = query / (self.d_model**0.25)
        K = key / (self.d_model**0.25)
        if mask is not None:
            mask = mask.bool()
        output = F.scaled_dot_product_attention(Q, K, value, attn_mask=mask)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = F.softmax(scores, dim=-1)
        return output, attn_weights


class MultiQueryAttention(nn.Module):
    """Multi-head queries paired with shared key/value projections.

    Each head has its own query subspace but all heads reuse the same key/value set,
    which lowers KV memory bandwidth (beneficial for decoder-only LLMs) while
    retaining the ability to model diverse query patterns.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size = query.size(0)
        Q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = self.k_proj(key).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        V = self.v_proj(value).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = (
            torch.matmul(attn_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(output)
        return output, attn_weights


class GroupedQueryAttention(nn.Module):
    """Generalization of multi-query attention where heads are partitioned into groups.

    A smaller `num_kv_heads` set of key/value projections is repeated across query
    head groups, balancing expressiveness and efficiency. Used in recent large
    language models (e.g., LLaMA 2/3) to scale to many heads without duplicating
    expensive KV tensors.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.kv_dim)
        self.v_proj = nn.Linear(d_model, self.kv_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size = query.size(0)
        Q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = K.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        V = V.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = (
            torch.matmul(attn_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(output)
        return output, attn_weights


class BiLevelAttention(nn.Module):
    """Two-level attention block for multi-scale context aggregation.

    A coarse branch with fewer heads captures broad structure, while a fine branch
    with more heads targets local detail; their projected outputs are summed. This
    pattern appears in hierarchical Transformers for vision or document modeling.
    """

    def __init__(self, d_model: int, coarse_heads: int = 4, fine_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.coarse_attn = MultiHeadAttention(d_model, coarse_heads)
        self.fine_attn = MultiHeadAttention(d_model, fine_heads)
        self.coarse_proj = nn.Linear(d_model, d_model)
        self.fine_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        coarse_out, _ = self.coarse_attn(query, key, value, mask)
        fine_out, _ = self.fine_attn(query, key, value, mask)
        output = self.coarse_proj(coarse_out) + self.fine_proj(fine_out)
        return output, None


def main():
    torch.manual_seed(42)
    batch_size, seq_len, d_model, num_heads = 2, 10, 128, 8

    x = torch.randn(batch_size, seq_len, d_model)
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)

    models = {
        "SoftAttention": SoftAttention(d_model),
        "HardAttention": HardAttention(d_model),
        "SelfAttention": SelfAttention(d_model),
        "CrossAttention": CrossAttention(d_model),
        "MultiHeadAttention": MultiHeadAttention(d_model, num_heads),
        "AdditiveAttention": AdditiveAttention(d_model),
        "MultiplicativeAttention": MultiplicativeAttention(d_model),
        "LocalAttention": LocalAttention(d_model, window_size=3),
        "FlashAttention": FlashAttention(d_model),
        "MultiQueryAttention": MultiQueryAttention(d_model, num_heads),
        "GroupedQueryAttention": GroupedQueryAttention(
            d_model, num_heads, num_kv_heads=4
        ),
        "BiLevelAttention": BiLevelAttention(d_model),
    }

    for name, model in models.items():
        try:
            if name == "SelfAttention":
                out, attn = model(x, mask)
            else:
                out, attn = model(q, k, v, mask)
            expected_shape = (batch_size, seq_len, d_model)
            assert out.shape == expected_shape, (
                f"{name}: Expected {expected_shape}, got {out.shape}"
            )
            print(f"{name}: Success - Output shape {out.shape}")
        except Exception as e:
            print(f"{name}: Failed - {str(e)}")


if __name__ == "__main__":
    main()
