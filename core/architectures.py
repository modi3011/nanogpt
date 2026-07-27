"""
Shared GPT-style decoder-only transformer architecture.

Improvements over the original nanoGPT:
  - RoPE (Rotary Position Embeddings) instead of learned absolute positional
    embeddings: better length generalisation, no wpe parameter cost.
  - Grouped Query Attention (GQA): fewer KV heads than Q heads, saving params
    in attention so they can be spent on depth/width instead.
  - Both the "stories" model (GPT-2 style: LayerNorm + GELU MLP) and the
    "code" model (LLaMA style: RMSNorm + SwiGLU MLP) use these improvements.
  - Variant is selected via GPTConfig.norm_type / mlp_type / n_kv_head.

References:
  GPT-2:    https://github.com/openai/gpt-2
  RoPE:     https://arxiv.org/abs/2104.09864
  GQA:      https://arxiv.org/abs/2305.13245
  LLaMA 2:  https://arxiv.org/abs/2307.09288
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch built-in requires bias=True)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation — no mean-centering, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def build_norm(norm_type: str, n_embd: int, bias: bool) -> nn.Module:
    """Factory that returns the configured normalisation layer."""
    if norm_type == "layernorm":
        return LayerNorm(n_embd, bias=bias)
    if norm_type == "rmsnorm":
        return RMSNorm(n_embd)
    raise ValueError(f"Unknown norm_type: {norm_type!r}  (expected 'layernorm' or 'rmsnorm')")


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension of x."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: tensors of shape (B, n_head, T, head_dim)
        cos, sin: precomputed tensors of shape (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE sin/cos tables up to `max_seq_len` positions."""

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        # inverse frequencies: one per pair of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # precompute tables and cache them as buffers (no grad)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)           # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)         # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for the first `seq_len` positions."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ---------------------------------------------------------------------------
# Attention (with GQA + RoPE)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with:
      - Flash Attention (PyTorch >= 2.0) or manual fallback
      - Grouped Query Attention (GQA): n_kv_head <= n_head
      - Rotary Position Embeddings (RoPE) on Q and K
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd    = config.n_embd
        self.dropout   = config.dropout
        self.head_dim  = config.n_embd // config.n_head
        self.n_rep     = config.n_head // config.n_kv_head  # KV repetitions for GQA

        # Separate Q / KV projections — Q is full-rank, KV uses fewer heads
        self.q_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention — Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                      .view(1, 1, config.block_size, config.block_size),
            )

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand KV heads to match the number of Q heads for GQA.

        Args:
            x:     (B, n_kv_head, T, head_dim)
            n_rep: repetition factor (n_head // n_kv_head)
        Returns:
            (B, n_head, T, head_dim)
        """
        if n_rep == 1:
            return x
        B, n_kv_head, T, hd = x.shape
        return (
            x[:, :, None, :, :]
             .expand(B, n_kv_head, n_rep, T, hd)
             .reshape(B, n_kv_head * n_rep, T, hd)
        )

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        """
        Args:
            x:    (B, T, C)
            rope: shared RotaryEmbedding module
        Returns:
            (B, T, C)
        """
        B, T, C = x.size()

        # Q: (B, n_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # K, V: (B, n_kv_head, T, head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        # apply RoPE to Q and K
        cos, sin = rope(T)
        q, k = apply_rope(q, k, cos, sin)

        # expand KV heads to match Q heads (GQA)
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# Feed-forward (MLP) variants
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """GPT-2-style feed-forward: Linear → GELU → Linear."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SwiGLU(nn.Module):
    """LLaMA-style SwiGLU feed-forward: silu(W1·x) ⊙ W2·x → W3.

    Hidden dimension is scaled to 2/3 × 4d and rounded to nearest multiple
    of 256 to keep it hardware-friendly, matching the LLaMA 2 approach.
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden = int(2 * 4 * config.n_embd / 3)
        # round up to nearest multiple of 256 for hardware efficiency
        hidden = 256 * ((hidden + 255) // 256)
        self.w1      = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2      = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3      = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def build_mlp(mlp_type: str, config: "GPTConfig") -> nn.Module:
    """Factory that returns the configured feed-forward block."""
    if mlp_type == "gelu":
        return MLP(config)
    if mlp_type == "swiglu":
        return SwiGLU(config)
    raise ValueError(f"Unknown mlp_type: {mlp_type!r}  (expected 'gelu' or 'swiglu')")


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → residual,
                                   LayerNorm → MLP       → residual."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = build_norm(config.norm_type, config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config.norm_type, config.n_embd, config.bias)
        self.mlp  = build_mlp(config.mlp_type, config)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), rope)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Full configuration for a GPT model.

    norm_type:  'layernorm' (GPT-2, default for stories) | 'rmsnorm' (LLaMA, default for code)
    mlp_type:   'gelu' (GPT-2 MLP)                       | 'swiglu'  (LLaMA gated MLP)
    n_kv_head:  number of KV heads for GQA; must divide n_head evenly.
                Set equal to n_head to disable GQA (standard MHA).
    """
    block_size: int   = 512
    vocab_size: int   = 50304    # GPT-2 vocab (50257) padded to nearest multiple of 64
    n_layer:    int   = 9
    n_head:     int   = 8
    n_kv_head:  int   = 2        # GQA: 2 KV heads shared across 8 Q heads
    n_embd:     int   = 480
    dropout:    float = 0.1
    bias:       bool  = True
    norm_type:  str   = "layernorm"
    mlp_type:   str   = "gelu"


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Decoder-only transformer language model with RoPE + GQA."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        head_dim = config.n_embd // config.n_head

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            # NOTE: no wpe — position information comes from RoPE
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = build_norm(config.norm_type, config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: token embedding ↔ output head
        self.transformer.wte.weight = self.lm_head.weight

        # one shared RoPE module for all blocks
        self.rope = RotaryEmbedding(head_dim, max_seq_len=config.block_size)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper §2.3)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w3.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count.

        When non_embedding=True (default), excludes the token embedding weight.
        The embedding is still counted in lm_head via weight tying, so this is
        consistent with the nanoGPT convention.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:     token indices of shape (B, T)
            targets: optional target indices of shape (B, T) for loss computation

        Returns:
            (logits, loss) — loss is None when targets is None.
            When targets is None (inference), logits has shape (B, 1, vocab_size)
            (only the last position is computed for efficiency).
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x, self.rope)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])   # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Shrink the model's maximum context length post-init."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # RoPE cache will rebuild lazily on the next forward pass if needed
        if block_size > self.rope.cos_cached.shape[0]:
            return
        self.rope.cos_cached = self.rope.cos_cached[:block_size]
        self.rope.sin_cached = self.rope.sin_cached[:block_size]

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay on 2D+ tensors only, fused when on CUDA."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"decayed param tensors: {len(decay_params)}, {n_decay:,} params")
        print(f"non-decayed tensors:   {len(nodecay_params)}, {n_nodecay:,} params")

        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **(dict(fused=True) if use_fused else {}),
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOP utilisation as fraction of A100 bf16 peak.
        (PaLM paper Appendix B: https://arxiv.org/abs/2204.02311)
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token    = 6 * N + 12 * L * H * Q * T
        flops_per_iter     = flops_per_token * T * fwdbwd_per_iter
        flops_achieved     = flops_per_iter / dt
        flops_promised     = 312e12   # A100 bf16 peak
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eot_token: Optional[int] = None,
        stop_at_second_eot: bool = False,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens.

        Args:
            idx:                conditioning tokens (B, T)
            max_new_tokens:     how many tokens to generate
            temperature:        softmax temperature (<1 = sharper, >1 = more random)
            top_k:              if set, restrict sampling to top-k logits
            eot_token:          stop early when this token is sampled
            stop_at_second_eot: (stories) run until the 2nd EOT and return the
                                span between them (the generated story body)

        Returns:
            Token tensor (B, T+generated), or just the story span when
            stop_at_second_eot is True and two EOTs were found.
        """
        eot_positions: list = []

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.item() == eot_token:
                eot_positions.append(idx.size(1) - 1)
                if stop_at_second_eot:
                    if len(eot_positions) == 2:
                        break
                else:
                    break

        if stop_at_second_eot:
            if len(eot_positions) >= 2:
                # clean case — return span between first and second EOT
                return idx[:, eot_positions[0] + 1 : eot_positions[1]]
            elif len(eot_positions) == 1:
                # only one EOT generated — return everything up to it
                return idx[:, eot_positions[0] + 1 :]

        return idx
