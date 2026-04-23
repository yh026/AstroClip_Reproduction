import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LayerNorm, TransformerBlock, init_linear_by_depth


@dataclass
class SpecFormerConfig:
    input_dim: int = 22
    embed_dim: int = 768
    num_layers: int = 6
    num_heads: int = 6
    max_len: int = 800
    dropout: float = 0.0
    mask_num_chunks: int = 6
    mask_chunk_width: int = 50
    slice_section_length: int = 20
    slice_overlap: int = 10


class SpecFormer(nn.Module):
    """
    Minimal standalone spectrum encoder/reconstructor derived from AstroCLIP's SpecFormer.

    Input expected before preprocess: (B, raw_length, 1) or (B, raw_length).
    Output after preprocess and slicing: (B, num_tokens, input_dim)
    where input_dim = 2 + slice_section_length.
    """

    def __init__(self, cfg: SpecFormerConfig):
        super().__init__()
        self.cfg = cfg

        self.data_embed = nn.Linear(cfg.input_dim, cfg.embed_dim)
        self.position_embed = nn.Embedding(cfg.max_len, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    causal=False,
                    bias=True,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_ln = LayerNorm(cfg.embed_dim, bias=True)
        self.head = nn.Linear(cfg.embed_dim, cfg.input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in [self.data_embed, self.position_embed]:
            std = 1.0 / math.sqrt(self.cfg.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)
            if hasattr(emb, "bias") and emb.bias is not None:
                nn.init.zeros_(emb.bias)

        self.blocks.apply(lambda m: init_linear_by_depth(m, self.cfg.num_layers))
        self.head.apply(lambda m: init_linear_by_depth(m, 0.5))

    def forward(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(spectrum)
        return self.forward_preprocessed(x)

    def encode(
        self,
        spectrum: torch.Tensor,
        *,
        preprocessed: bool = False,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        x = spectrum if preprocessed else self.preprocess(spectrum)
        return self.forward_preprocessed(x, return_all_layers=return_all_layers)

    def forward_preprocessed(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        bsz, seqlen, _ = x.shape
        if seqlen > self.cfg.max_len:
            raise ValueError(
                f"Sequence length {seqlen} exceeds max_len={self.cfg.max_len}. "
                "Increase max_len or change slice parameters."
            )

        pos = torch.arange(0, seqlen, dtype=torch.long, device=x.device)
        token_embed = self.data_embed(x) + self.position_embed(pos)
        h = self.dropout(token_embed)

        all_layers: List[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            if return_all_layers:
                all_layers.append(h)

        embedding = self.final_ln(h)
        recon = self.head(embedding)

        outputs: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "token_embedding": token_embed,
            "embedding": embedding,
            "reconstructions": recon,
        }
        if return_all_layers:
            outputs["all_layer_embeddings"] = all_layers
        return outputs

    def compute_loss(self, spectrum: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        target = self.preprocess(spectrum)
        masked_input = self.mask_sequence(target)
        outputs = self.forward_preprocessed(masked_input)
        recon = outputs["reconstructions"]

        mask_locs = (masked_input != target).type_as(recon)
        masked_fraction = mask_locs.mean().clamp_min(1e-8)
        loss = F.mse_loss(recon * mask_locs, target * mask_locs, reduction="mean") / masked_fraction

        stats = {
            "loss": float(loss.detach().cpu()),
            "masked_fraction": float(masked_fraction.detach().cpu()),
        }
        cache = {
            "target": target.detach(),
            "masked_input": masked_input.detach(),
            "reconstructions": recon.detach(),
            "mask_locs": mask_locs.detach(),
        }
        return loss, stats, cache

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (B, L) or (B, L, 1), got {tuple(x.shape)}")

        x = x.float()
        std = x.std(dim=1, keepdim=True).clamp_min_(0.2)
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean) / std
        x = self.slice_spectrum(x)

        # prepend one extra token row and two feature channels.
        # channel 0/1 in first token carry sample-level mean/std summaries.
        x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0.0)
        x[:, 0, 0] = (mean.squeeze(-1).squeeze(-1) - 2.0) / 2.0
        x[:, 0, 1] = (std.squeeze(-1).squeeze(-1) - 2.0) / 8.0
        return x

    def slice_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        raw_len = x.shape[1]
        step = self.cfg.slice_section_length - self.cfg.slice_overlap
        starts = np.arange(0, raw_len - self.cfg.slice_overlap, step)
        sections = [x[:, s : s + self.cfg.slice_section_length].transpose(1, 2) for s in starts]
        if not sections:
            raise ValueError("No spectrum slices produced. Check input length and slice params.")
        if sections[-1].shape[2] < self.cfg.slice_section_length:
            sections.pop(-1)
        if not sections:
            raise ValueError("Input became empty after dropping the last short slice.")
        return torch.cat(sections, dim=1)

    def mask_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.mask_one_sequence(seq) for seq in x], dim=0)

    def mask_one_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        seq_len = seq.shape[0]
        num_chunks = self.cfg.mask_num_chunks
        chunk_width = self.cfg.mask_chunk_width

        total_width_needed = num_chunks * chunk_width + (num_chunks - 1) * chunk_width
        if total_width_needed > seq_len:
            raise ValueError(
                f"Sequence too short for masking: len={seq_len}, need at least {total_width_needed}. "
                "Lower mask_num_chunks / mask_chunk_width or increase number of tokens."
            )

        out = seq.clone()
        for i in range(num_chunks):
            band_start = (i * seq_len) // num_chunks
            local_range = seq_len // num_chunks - chunk_width
            if local_range <= 0:
                raise ValueError("Mask chunk width is too large for one partition.")
            offset = torch.randint(0, local_range, (1,), device=seq.device).item()
            out[band_start + offset : band_start + offset + chunk_width] = 0.0
        return out
