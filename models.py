"""Decoder models for SemComm.

  ImageDecoder        z (B, 1024) -> image (B, 3, 224, 224) in [-1, 1]
  TransformerMapper   z (B, 1024) -> prefix tokens (B, K, d_model)   (ClipCap-style)
  MLPMapper           same I/O, simpler alternative
  TextDecoder         frozen Qwen3-0.6B-Base + mapper, exposes forward (CE) and generate

Inputs are assumed already unit-norm (encoder side ensures this).
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn


# 학습 시 EOS를 못 본 모델은 max_new_tokens 끝까지 채우며 LM pretraining 패턴
# (_, <br />, 반복문장 등)을 흘림. inference에선 첫 문장만 살린다.
_SENT_END = re.compile(r"^(.*?[.!?])(?:\s|$)")

def _clean_caption(s: str) -> str:
    s = s.strip()
    m = _SENT_END.match(s)
    return m.group(1).strip() if m else s


# ---------------------------------------------------------------------------
# Image decoder

class ImageDecoder(nn.Module):
    """Simple ConvTranspose decoder. Output range [-1, 1] (tanh)."""

    def __init__(self, z_dim: int = 1024, hidden_init: int = 512):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_init = hidden_init
        self.proj = nn.Linear(z_dim, 7 * 7 * hidden_init)

        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        ch = [hidden_init, 256, 128, 64, 32, 3]
        layers: list[nn.Module] = []
        for i in range(5):
            in_c, out_c = ch[i], ch[i + 1]
            layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            if i < 4:
                groups = 8 if out_c % 8 == 0 else 1
                layers.append(nn.GroupNorm(groups, out_c))
                layers.append(nn.SiLU(inplace=True))
            else:
                layers.append(nn.Tanh())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(z.size(0), self.hidden_init, 7, 7)
        return self.deconv(h)


# ---------------------------------------------------------------------------
# Mappers (z -> prefix tokens)

class TransformerMapper(nn.Module):
    """ClipCap-style mapper for frozen LM. Larger capacity than MLP, recommended
    when the LM doesn't co-adapt to the prefix (i.e. freeze_lm=True)."""

    def __init__(self, z_dim: int, lm_dim: int, prefix_len: int,
                 num_layers: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        self.linear = nn.Linear(z_dim, prefix_len * lm_dim)
        # Learnable constant queries, mixed with linearly-projected z via self-attention.
        self.const = nn.Parameter(torch.randn(prefix_len, lm_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lm_dim, nhead=num_heads, dim_feedforward=4 * lm_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        proj = self.linear(z).view(B, self.prefix_len, self.lm_dim)
        const = self.const.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([proj, const], dim=1)               # (B, 2K, d)
        out = self.transformer(x)                          # (B, 2K, d)
        return out[:, : self.prefix_len]                  # (B, K, d)


class MLPMapper(nn.Module):
    def __init__(self, z_dim: int, lm_dim: int, prefix_len: int,
                 hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        hidden = hidden or (z_dim + prefix_len * lm_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, prefix_len * lm_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.size(0), self.prefix_len, self.lm_dim)


def build_mapper(mapper_type: str, z_dim: int, lm_dim: int, prefix_len: int,
                 num_layers: int, num_heads: int, dropout: float) -> nn.Module:
    if mapper_type == "transformer":
        return TransformerMapper(z_dim, lm_dim, prefix_len, num_layers, num_heads, dropout)
    if mapper_type == "mlp":
        return MLPMapper(z_dim, lm_dim, prefix_len, dropout=dropout)
    raise ValueError(f"unknown mapper_type: {mapper_type}")


# ---------------------------------------------------------------------------
# Text decoder = frozen LM + mapper

class TextDecoder(nn.Module):
    """ClipCap-style text decoder built on a HuggingFace causal LM.

    Training: forward(z, input_ids, attention_mask) -> CE loss (labels auto-shifted by HF).
              Loss is masked at prefix positions via labels=-100.
    Inference: generate(z, beam_size, max_new_tokens) -> list[str].
    """

    def __init__(self, lm_name: str, z_dim: int = 1024, prefix_len: int = 10,
                 mapper_type: str = "transformer", mapper_layers: int = 8,
                 mapper_heads: int = 8, mapper_dropout: float = 0.1,
                 freeze_lm: bool = True, lm_dtype: str = "auto"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype_map = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
        if lm_dtype == "auto":
            torch_dtype = (torch.bfloat16
                           if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                           else torch.float32)
        else:
            torch_dtype = dtype_map[lm_dtype]
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.freeze_lm = freeze_lm
        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
            self.lm.eval()

        d = self.lm.config.hidden_size
        self.lm_dim = d
        self.prefix_len = prefix_len
        self.mapper = build_mapper(
            mapper_type, z_dim, d, prefix_len,
            mapper_layers, mapper_heads, mapper_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen LM in eval mode (no dropout/BN drift).
        if self.freeze_lm:
            self.lm.eval()
        return self

    def trainable_parameters(self):
        return self.mapper.parameters()

    def get_token_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(input_ids)

    def forward(self, z: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute LM cross-entropy, with prefix positions masked out (-100)."""
        B, T = input_ids.shape
        prefix = self.mapper(z)                                                  # (B, K, d)
        token_embeds = self.get_token_embeds(input_ids)                          # (B, T, d)
        # Match dtype of LM (e.g., fp16/bf16 weights yield same-dtype embeddings;
        # mapper is fp32 by default → cast prefix to match for concat).
        prefix = prefix.to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)                 # (B, K+T, d)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(B, self.prefix_len, device=z.device,
                                 dtype=attention_mask.dtype)
        full_attn = torch.cat([prefix_mask, attention_mask], dim=1)              # (B, K+T)

        # Loss only on token positions; pad positions also masked.
        ignore = -100
        prefix_labels = torch.full((B, self.prefix_len), ignore,
                                   device=z.device, dtype=input_ids.dtype)
        token_labels = input_ids.masked_fill(attention_mask == 0, ignore)
        labels = torch.cat([prefix_labels, token_labels], dim=1)                 # (B, K+T)

        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=full_attn,
                      labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, z: torch.Tensor, beam_size: int = 5,
                 max_new_tokens: int = 30, clean: bool = True) -> list[str]:
        """Generate captions from z using HuggingFace generate(inputs_embeds=...).

        Returns only the newly-generated tokens decoded as text (HF convention
        when inputs_embeds is the prompt). If clean=True (default), trims to
        the first sentence — needed when the LM was trained without EOS in
        captions and pads with pretraining junk past the natural sentence end.
        """
        B = z.size(0)
        prefix = self.mapper(z)                                                  # (B, K, d)
        # Cast to LM dtype.
        lm_dtype = next(self.lm.parameters()).dtype
        prefix = prefix.to(lm_dtype)
        attn = torch.ones(B, self.prefix_len, device=z.device, dtype=torch.long)

        gen_ids = self.lm.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [_clean_caption(t) for t in texts] if clean else texts
