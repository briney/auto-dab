"""
Antibody Masked Diffusion Language Model -- single-file training.

This file contains the complete model, diffusion, and training loop.
It is the ONLY file the autoresearch agent modifies.

Usage:
    uv run train.py
"""

from __future__ import annotations

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

import prepare

# ============================================================================
# Hyperparameters (all editable by the agent)
# ============================================================================

# -- Model --
DEPTH = 24
D_MODEL = 256
N_HEADS = 4
FFN_MULTIPLIER = 8 / 3
NORM_TYPE = "rmsnorm"  # "layernorm" | "rmsnorm"
USE_CHAIN_AWARE_ATTENTION = True
QK_NORM = "none"  # "none" | "norm" | "learned_scale"
DROPOUT = 0.0

# -- Diffusion --
NOISE_SCHEDULE = "cosine"  # "linear" | "cosine" | "sqrt" | "power" | "static"
NUM_TIMESTEPS = 100
POWER = 4.0  # only for power schedule
STATIC_MASK_RATE = 0.15  # only for static schedule
MASKING = "uniform"  # "uniform" | "information_weighted"
CDR_WEIGHT = 1.0
NONGERMLINE_WEIGHT = 1.0
SELECTION_METHOD = "sampled"  # "sampled" | "ranked"
LOSS_OBJECTIVE = "mlm"  # "mlm" | "nelbo"
NELBO_NORMALIZE = None  # None | "clip" | "minmax"
NELBO_CLIP_MAX = 10.0
USE_CURRICULUM = False
CURRICULUM_START = 0.1

# -- Training --
TOTAL_BATCH_SIZE = 2**15  # ~32K tokens
DEVICE_BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_FRAC = 0.05
WARMDOWN_FRAC = 0.3
FINAL_LR_FRAC = 0.0
SEED = 42
WARMUP_STEPS = 10  # steps excluded from time budget (torch.compile warmup)


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def create_norm(dim: int, norm_type: str = NORM_TYPE, eps: float = 1e-6) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps)
    return nn.LayerNorm(dim, eps=eps)


class LearnedQKScale(nn.Module):
    """Per-head learned scaling for Q and K."""

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.k_scale = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return q * self.q_scale, k * self.k_scale


class QKNormModule(nn.Module):
    """Applies normalization to Q and K tensors."""

    def __init__(self, head_dim: int, norm_type: str = NORM_TYPE) -> None:
        super().__init__()
        self.q_norm = create_norm(head_dim, norm_type)
        self.k_norm = create_norm(head_dim, norm_type)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.q_norm(q), self.k_norm(k)


def create_qk_norm(
    qk_norm: str, n_heads: int, head_dim: int
) -> nn.Module | None:
    if qk_norm == "none":
        return None
    if qk_norm == "norm":
        return QKNormModule(head_dim)
    if qk_norm == "learned_scale":
        return LearnedQKScale(n_heads)
    raise ValueError(f"Unknown qk_norm: {qk_norm}")


# ============================================================================
# Rotary Position Embedding
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    """RoPE for transformer attention."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


# ============================================================================
# Feed-Forward Network
# ============================================================================

class FusedSwiGLUFFN(nn.Module):
    """Memory-efficient SwiGLU FFN with fused gate/up projection."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, d_ffn * 2, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.w_down(self.dropout(F.silu(gate) * up))


# ============================================================================
# Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with RoPE and SDPA."""

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, dropout: float = 0.0,
        max_seq_len: int = 512, qk_norm: str = "none",
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.dropout_p = dropout
        inner_dim = n_heads * head_dim

        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)

        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)
        self.qk_norm_module = create_qk_norm(qk_norm, n_heads, head_dim)

    def _make_padding_mask(self, attention_mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
        if attention_mask is None:
            return None
        mask = torch.zeros_like(attention_mask, dtype=dtype)
        mask = mask.masked_fill(~attention_mask.bool(), float("-inf"))
        return mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    def forward(
        self, x: Tensor, chain_ids: Tensor, attention_mask: Tensor | None = None,
    ) -> Tensor:
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_heads)

        q, k = self.rope(q, k)
        if self.qk_norm_module is not None:
            q, k = self.qk_norm_module(q, k)

        padding_mask = self._make_padding_mask(attention_mask, x.dtype)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=padding_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
        )
        return self.out_proj(rearrange(output, "b h s d -> b s (h d)"))


class ChainAwareAttention(nn.Module):
    """MINT-style hybrid intra/inter-chain attention.

    Intra-chain pairs use self-attention scores (with RoPE).
    Inter-chain pairs use cross-attention scores (no RoPE).
    Single merged softmax, then split routing to values.
    """

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, dropout: float = 0.0,
        max_seq_len: int = 512, qk_norm: str = "none",
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner_dim = n_heads * head_dim

        # Self-attention path (RoPE applied)
        self.q_self = nn.Linear(d_model, inner_dim, bias=False)
        self.k_self = nn.Linear(d_model, inner_dim, bias=False)
        self.v_self = nn.Linear(d_model, inner_dim, bias=False)

        # Cross-attention path (no RoPE)
        self.q_cross = nn.Linear(d_model, inner_dim, bias=False)
        self.k_cross = nn.Linear(d_model, inner_dim, bias=False)
        self.v_cross = nn.Linear(d_model, inner_dim, bias=False)

        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.qk_norm_self = create_qk_norm(qk_norm, n_heads, head_dim)
        self.qk_norm_cross = create_qk_norm(qk_norm, n_heads, head_dim)

    def _make_padding_mask(self, attention_mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
        if attention_mask is None:
            return None
        mask = torch.zeros_like(attention_mask, dtype=dtype)
        mask = mask.masked_fill(~attention_mask.bool(), float("-inf"))
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self, x: Tensor, chain_ids: Tensor, attention_mask: Tensor | None = None,
    ) -> Tensor:
        B, S, _ = x.shape
        h = self.n_heads

        # Self-attention projections + RoPE
        q_s = rearrange(self.q_self(x), "b s (h d) -> b h s d", h=h)
        k_s = rearrange(self.k_self(x), "b s (h d) -> b h s d", h=h)
        v_s = rearrange(self.v_self(x), "b s (h d) -> b h s d", h=h)
        q_s, k_s = self.rope(q_s, k_s)

        # Cross-attention projections (no RoPE)
        q_c = rearrange(self.q_cross(x), "b s (h d) -> b h s d", h=h)
        k_c = rearrange(self.k_cross(x), "b s (h d) -> b h s d", h=h)
        v_c = rearrange(self.v_cross(x), "b s (h d) -> b h s d", h=h)

        # QK normalization
        if self.qk_norm_self is not None:
            q_s, k_s = self.qk_norm_self(q_s, k_s)
        if self.qk_norm_cross is not None:
            q_c, k_c = self.qk_norm_cross(q_c, k_c)

        # Attention scores
        scores_self = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
        scores_cross = torch.matmul(q_c, k_c.transpose(-2, -1)) * self.scale

        # Chain mask: True where same chain (intra-chain)
        chain_i = chain_ids.unsqueeze(-1)  # (B, S, 1)
        chain_j = chain_ids.unsqueeze(-2)  # (B, 1, S)
        intra_mask = (chain_i == chain_j).unsqueeze(1)  # (B, 1, S, S)

        # Merge: self scores for intra-chain, cross scores for inter-chain
        merged = torch.where(intra_mask, scores_self, scores_cross)

        # Padding mask
        padding_mask = self._make_padding_mask(attention_mask, x.dtype)
        if padding_mask is not None:
            merged = merged + padding_mask

        # Single softmax
        attn_weights = F.softmax(merged, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Split routing: intra -> v_self, inter -> v_cross
        intra_float = intra_mask.to(x.dtype)
        out = torch.matmul(attn_weights * intra_float, v_s) + \
              torch.matmul(attn_weights * (1.0 - intra_float), v_c)

        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


# ============================================================================
# Transformer
# ============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + SwiGLU FFN."""

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, d_ffn: int,
        dropout: float = 0.0, max_seq_len: int = 512,
        use_chain_aware: bool = True, qk_norm: str = "none",
    ) -> None:
        super().__init__()
        self.attn_norm = create_norm(d_model)
        self.ffn_norm = create_norm(d_model)

        attn_cls = ChainAwareAttention if use_chain_aware else MultiHeadAttention
        self.attention = attn_cls(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim,
            dropout=dropout, max_seq_len=max_seq_len, qk_norm=qk_norm,
        )
        self.ffn = FusedSwiGLUFFN(d_model, d_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, chain_ids: Tensor, attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Attention sublayer (pre-norm)
        x = x + self.dropout(self.attention(self.attn_norm(x), chain_ids, attention_mask))
        # FFN sublayer (pre-norm)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DAbModel(nn.Module):
    """Discrete Diffusion Antibody Language Model."""

    def __init__(
        self,
        vocab_size: int = prepare.VOCAB_SIZE,
        d_model: int = D_MODEL,
        n_layers: int = DEPTH,
        n_heads: int = N_HEADS,
        ffn_multiplier: float = FFN_MULTIPLIER,
        dropout: float = DROPOUT,
        max_seq_len: int = prepare.MAX_SEQ_LEN,
        use_chain_aware: bool = USE_CHAIN_AWARE_ATTENTION,
        qk_norm: str = QK_NORM,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        head_dim = d_model // n_heads
        d_ffn = int(d_model * ffn_multiplier)
        d_ffn = ((d_ffn + 63) // 64) * 64  # round to multiple of 64

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=prepare.PAD_TOKEN_ID)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, head_dim=head_dim,
                d_ffn=d_ffn, dropout=dropout, max_seq_len=max_seq_len,
                use_chain_aware=use_chain_aware, qk_norm=qk_norm,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = create_norm(d_model)

        # LM head (tied with token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self, token_ids: Tensor, chain_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        x = self.token_embedding(token_ids) * self.embed_scale
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, chain_ids, attention_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return {"logits": logits}

    def get_num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.token_embedding.weight.numel()  # tied with lm_head
        return n


# ============================================================================
# Noise Schedules
# ============================================================================

class NoiseSchedule:
    """Base noise schedule."""

    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        raise NotImplementedError

    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        raise NotImplementedError

    def sample_timesteps(
        self, batch_size: int, device: torch.device,
        training_progress: float | None = None,
    ) -> Tensor:
        if training_progress is not None and training_progress < 1.0:
            frac = CURRICULUM_START + (1.0 - CURRICULUM_START) * training_progress
            max_t = max(1, int(self.num_timesteps * frac))
        else:
            max_t = self.num_timesteps
        return torch.randint(1, max_t + 1, (batch_size,), device=device)


class LinearSchedule(NoiseSchedule):
    def get_mask_rate(self, t: int | Tensor) -> float | Tensor:
        return t / self.num_timesteps

    def get_nelbo_weight(self, t: int | Tensor) -> float | Tensor:
        if isinstance(t, Tensor):
            return self.num_timesteps / t.float()
        return self.num_timesteps / t


class CosineSchedule(NoiseSchedule):
    def get_mask_rate(self, t: int | Tensor) -> float | Tensor:
        t_norm = t / self.num_timesteps
        if isinstance(t_norm, Tensor):
            return 1 - torch.cos(t_norm * math.pi / 2)
        return 1 - math.cos(t_norm * math.pi / 2)

    def get_nelbo_weight(self, t: int | Tensor) -> float | Tensor:
        t_norm = t / self.num_timesteps
        if isinstance(t_norm, Tensor):
            t_norm = t_norm.float()
            sin_t = torch.sin(t_norm * math.pi / 2)
            cos_t = torch.cos(t_norm * math.pi / 2)
            denom = (1 - cos_t).clamp(min=1e-8)
            return (math.pi / (2 * self.num_timesteps)) * sin_t / denom
        sin_t = math.sin(t_norm * math.pi / 2)
        cos_t = math.cos(t_norm * math.pi / 2)
        return (math.pi / (2 * self.num_timesteps)) * sin_t / max(1 - cos_t, 1e-8)


class SqrtSchedule(NoiseSchedule):
    def get_mask_rate(self, t: int | Tensor) -> float | Tensor:
        t_norm = t / self.num_timesteps
        if isinstance(t_norm, Tensor):
            return torch.sqrt(t_norm)
        return math.sqrt(t_norm)

    def get_nelbo_weight(self, t: int | Tensor) -> float | Tensor:
        if isinstance(t, Tensor):
            return self.num_timesteps / (2 * t.float())
        return self.num_timesteps / (2 * t)


class PowerSchedule(NoiseSchedule):
    def __init__(self, num_timesteps: int, power: float = 4.0) -> None:
        super().__init__(num_timesteps)
        self.power = power

    def get_mask_rate(self, t: int | Tensor) -> float | Tensor:
        t_norm = t / self.num_timesteps
        if isinstance(t_norm, Tensor):
            return torch.pow(t_norm, self.power)
        return t_norm**self.power

    def get_nelbo_weight(self, t: int | Tensor) -> float | Tensor:
        if isinstance(t, Tensor):
            return self.power * self.num_timesteps / t.float()
        return self.power * self.num_timesteps / t


class StaticSchedule(NoiseSchedule):
    def __init__(self, num_timesteps: int, mask_rate: float = 0.15) -> None:
        super().__init__(num_timesteps)
        self.mask_rate = mask_rate

    def get_mask_rate(self, t: int | Tensor) -> float | Tensor:
        if isinstance(t, Tensor):
            return torch.full_like(t, self.mask_rate, dtype=torch.float)
        return self.mask_rate

    def get_nelbo_weight(self, t: int | Tensor) -> float | Tensor:
        if isinstance(t, Tensor):
            return torch.ones_like(t, dtype=torch.float)
        return 1.0


def create_schedule() -> NoiseSchedule:
    if NOISE_SCHEDULE == "linear":
        return LinearSchedule(NUM_TIMESTEPS)
    if NOISE_SCHEDULE == "cosine":
        return CosineSchedule(NUM_TIMESTEPS)
    if NOISE_SCHEDULE == "sqrt":
        return SqrtSchedule(NUM_TIMESTEPS)
    if NOISE_SCHEDULE == "power":
        return PowerSchedule(NUM_TIMESTEPS, POWER)
    if NOISE_SCHEDULE == "static":
        return StaticSchedule(NUM_TIMESTEPS, STATIC_MASK_RATE)
    raise ValueError(f"Unknown noise schedule: {NOISE_SCHEDULE}")


# ============================================================================
# Masking
# ============================================================================

def apply_uniform_mask(
    token_ids: Tensor, timesteps: Tensor, attention_mask: Tensor,
    special_tokens_mask: Tensor, schedule: NoiseSchedule,
) -> tuple[Tensor, Tensor]:
    """Apply uniform random masking."""
    B, S = token_ids.shape
    device = token_ids.device

    mask_rates = schedule.get_mask_rate(timesteps)
    rand = torch.rand(B, S, device=device)
    maskable = attention_mask.bool() & ~special_tokens_mask
    mask_labels = (rand < mask_rates.unsqueeze(-1)) & maskable

    masked_ids = token_ids.clone()
    masked_ids[mask_labels] = prepare.MASK_TOKEN_ID
    return masked_ids, mask_labels


def apply_information_weighted_mask(
    token_ids: Tensor, timesteps: Tensor, attention_mask: Tensor,
    special_tokens_mask: Tensor, schedule: NoiseSchedule,
    cdr_mask: Tensor | None = None, non_templated_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply information-weighted masking (CDR/nongermline bias)."""
    B, S = token_ids.shape
    device = token_ids.device

    maskable = attention_mask.bool() & ~special_tokens_mask

    # Compute weights
    weights = torch.ones(B, S, device=device)
    if cdr_mask is not None:
        weights = weights + (cdr_mask > 0).float() * CDR_WEIGHT
    if non_templated_mask is not None:
        weights = weights + non_templated_mask.float() * NONGERMLINE_WEIGHT

    weights = weights * maskable.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # How many to mask per sample
    mask_rates = schedule.get_mask_rate(timesteps)
    valid_counts = maskable.sum(dim=-1)
    num_to_mask = (valid_counts.float() * mask_rates).round().long().clamp(min=0)

    # Gumbel-top-k or ranked selection
    if SELECTION_METHOD == "sampled":
        eps = 1e-10
        uniform = torch.rand(B, S, device=device).clamp(min=eps, max=1 - eps)
        gumbel = -torch.log(-torch.log(uniform))
        scores = torch.log(weights + eps) + gumbel
    else:
        noise = torch.rand(B, S, device=device) * 1e-6
        scores = weights + noise

    scores = scores.masked_fill(~maskable, float("-inf"))
    _, indices = scores.sort(dim=-1, descending=True)

    position_ranks = torch.zeros_like(indices)
    position_ranks.scatter_(
        dim=-1, index=indices,
        src=torch.arange(S, device=device).expand(B, -1),
    )

    mask_labels = (position_ranks < num_to_mask.unsqueeze(-1)) & maskable

    masked_ids = token_ids.clone()
    masked_ids[mask_labels] = prepare.MASK_TOKEN_ID
    return masked_ids, mask_labels


# ============================================================================
# Loss
# ============================================================================

def compute_loss(
    logits: Tensor, targets: Tensor, mask_labels: Tensor,
    timestep_weights: Tensor | None = None,
) -> Tensor:
    """Compute masked cross-entropy loss, optionally NELBO-weighted."""
    B, S, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)
    mask_flat = mask_labels.view(-1)

    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    if timestep_weights is not None:
        # NELBO: per-sample weighted loss
        loss_per_token = loss_per_token.view(B, S)
        masked_loss = loss_per_token * mask_labels.float()
        tokens_per_sample = mask_labels.sum(dim=1).clamp(min=1)
        sample_losses = masked_loss.sum(dim=1) / tokens_per_sample
        return (sample_losses * timestep_weights).mean()

    # Standard MLM loss: mean over masked positions
    masked_loss = loss_per_token * mask_flat.float()
    return masked_loss.sum() / mask_flat.sum().clamp(min=1)


# ============================================================================
# Learning Rate Schedule (time-based, Karpathy-style)
# ============================================================================

def get_lr(progress: float) -> float:
    """Compute learning rate multiplier based on wall-clock progress [0, 1]."""
    if progress < WARMUP_FRAC:
        # Linear warmup
        return progress / max(WARMUP_FRAC, 1e-8)
    elif progress > 1.0 - WARMDOWN_FRAC:
        # Cosine warmdown
        warmdown_progress = (progress - (1.0 - WARMDOWN_FRAC)) / max(WARMDOWN_FRAC, 1e-8)
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (1.0 + math.cos(math.pi * warmdown_progress))
    else:
        return 1.0


# ============================================================================
# Training
# ============================================================================

def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = DAbModel()
    model = model.to(device)
    num_params = model.get_num_params()
    print(f"Parameters: {num_params / 1e6:.1f}M (non-embedding)")

    # Compile
    model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )

    # Data
    grad_accum_steps = max(1, TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * prepare.MAX_SEQ_LEN))
    # For antibody sequences (short), batch in sequences not tokens
    # Use DEVICE_BATCH_SIZE directly as number of sequences per micro-batch
    train_loader = prepare.make_dataloader(DEVICE_BATCH_SIZE, split="train")
    train_iter = iter(train_loader)

    # Diffusion
    schedule = create_schedule()

    # Training state
    step = 0
    training_time = 0.0
    ema_loss = 0.0
    start_time = None

    print(f"Batch size: {DEVICE_BATCH_SIZE} seqs x {grad_accum_steps} accum steps")
    print(f"Noise schedule: {NOISE_SCHEDULE}, timesteps: {NUM_TIMESTEPS}")
    print(f"Masking: {MASKING}, loss: {LOSS_OBJECTIVE}")
    print(f"Time budget: {prepare.TIME_BUDGET}s")
    print("Starting training...")

    # GC management (Karpathy-style)
    gc.collect()
    torch.cuda.empty_cache()

    while True:
        t_step_start = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            # Get next batch (cycle through data)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            token_ids = batch["token_ids"].to(device)
            chain_ids = batch["chain_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)

            B = token_ids.shape[0]

            # Sample timesteps
            progress = training_time / prepare.TIME_BUDGET if USE_CURRICULUM else None
            timesteps = schedule.sample_timesteps(B, device, progress)

            # Apply masking
            if MASKING == "information_weighted":
                cdr = batch.get("cdr_mask")
                nt = batch.get("non_templated_mask")
                cdr = cdr.to(device) if cdr is not None else None
                nt = nt.to(device) if nt is not None else None
                masked_ids, mask_labels = apply_information_weighted_mask(
                    token_ids, timesteps, attention_mask,
                    special_tokens_mask, schedule, cdr, nt,
                )
            else:
                masked_ids, mask_labels = apply_uniform_mask(
                    token_ids, timesteps, attention_mask,
                    special_tokens_mask, schedule,
                )

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(masked_ids, chain_ids, attention_mask)
                logits = outputs["logits"]

                # Loss
                if LOSS_OBJECTIVE == "nelbo":
                    weights = schedule.get_nelbo_weight(timesteps)
                    if isinstance(weights, Tensor):
                        weights = weights.float()
                    if NELBO_NORMALIZE == "clip":
                        weights = weights.clamp(max=NELBO_CLIP_MAX)
                    elif NELBO_NORMALIZE == "minmax":
                        w_min, w_max = weights.min(), weights.max()
                        if (w_max - w_min) > 1e-8:
                            weights = (weights - w_min) / (w_max - w_min)
                    loss = compute_loss(logits, token_ids, mask_labels, weights)
                else:
                    loss = compute_loss(logits, token_ids, mask_labels)

                loss = loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping + optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        step += 1
        t_step_end = time.time()
        step_time = t_step_end - t_step_start

        # Warmup: first N steps don't count toward time budget
        if step <= WARMUP_STEPS:
            if step == WARMUP_STEPS:
                print(f"Warmup complete ({step} steps). Starting time budget.")
                start_time = time.time()
                gc.collect()
                torch.cuda.empty_cache()
            continue

        if start_time is None:
            start_time = time.time()

        training_time = time.time() - start_time

        # Update learning rate based on wall-clock progress
        progress = min(training_time / prepare.TIME_BUDGET, 1.0)
        lr_mult = get_lr(progress)
        for pg in optimizer.param_groups:
            pg["lr"] = LEARNING_RATE * lr_mult

        # EMA loss for logging
        ema_loss = 0.9 * ema_loss + 0.1 * accum_loss if ema_loss > 0 else accum_loss

        # Fast-fail
        if math.isnan(accum_loss) or accum_loss > 100:
            print(f"DIVERGED at step {step}: loss={accum_loss:.4f}")
            return

        # Log periodically
        if step % 50 == 0:
            print(
                f"step={step} loss={ema_loss:.4f} lr={LEARNING_RATE * lr_mult:.2e} "
                f"time={training_time:.1f}s/{prepare.TIME_BUDGET}s"
            )

        # GC every 5000 steps
        if step % 5000 == 0:
            gc.collect()

        # Time's up
        if training_time >= prepare.TIME_BUDGET:
            break

    total_time = time.time() - start_time if start_time else 0.0

    # Evaluate
    print("\nEvaluating...")
    results = prepare.evaluate(model, batch_size=DEVICE_BATCH_SIZE)

    # Peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    # Print structured summary
    print("\n" + "=" * 50)
    print(f"val_loss: {results['val_loss']:.6f}")
    print(f"val_accuracy: {results['val_accuracy']:.4f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds: {total_time:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {num_params / 1e6:.1f}")
    print(f"depth: {DEPTH}")
    print(f"d_model: {D_MODEL}")
    print(f"noise_schedule: {NOISE_SCHEDULE}")
    print(f"masking: {MASKING}")
    print(f"loss_objective: {LOSS_OBJECTIVE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
