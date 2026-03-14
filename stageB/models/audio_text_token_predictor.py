from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = int(d_model)
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, length: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe = torch.zeros(length, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            self.pe = self._build_pe(int(x.shape[1])).to(device=x.device)
        return x + self.pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)


class RhythmEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1, kernel_size: int = 5, n_conv_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        conv_layers = []
        for _ in range(int(n_conv_layers)):
            conv_layers.extend(
                [
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.conv = nn.Sequential(*conv_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h)


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        scalar_dim: int,
        d_model: int,
        bert_dim: int = 0,
        word_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for the bidirectional GRU TextEncoder, got {d_model}")
        self.bert_dim = int(bert_dim)
        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, word_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.bert_dim > 0:
            self.bert_proj = nn.Sequential(
                nn.Linear(self.bert_dim, word_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.bert_proj = None
        mix_in_dim = word_dim * (3 if self.bert_proj is not None else 2)
        self.mix = nn.Linear(mix_in_dim, d_model)
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, word_ids: torch.Tensor, scalar_feat: torch.Tensor, bert_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        word_feat = self.word_emb(word_ids)
        scalar_feat = self.scalar_proj(scalar_feat)
        feats = [word_feat, scalar_feat]
        if self.bert_proj is not None:
            if bert_feat is None:
                bert_feat = torch.zeros((*word_ids.shape, self.bert_dim), device=word_ids.device, dtype=word_feat.dtype)
            bert_feat = self.bert_proj(bert_feat.to(dtype=word_feat.dtype))
            feats.append(bert_feat)
        mixed = self.mix(torch.cat(feats, dim=-1))
        out, _ = self.gru(mixed)
        return self.norm(out)


class ContentRhythmFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.rhythm_proj = nn.Linear(d_model, d_model)
        self.content_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rhythm: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rhythm_proj(rhythm)
        c = self.content_proj(content)
        gate = torch.sigmoid(self.gate(torch.cat([r, c], dim=-1)))
        fused = gate * c + (1.0 - gate) * r
        fused = self.norm(fused + self.out_proj(fused))
        return fused, gate


class CodeHintEncoder(nn.Module):
    def __init__(self, codebook_sizes: Dict[str, int], d_model: int, hint_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.part_names = list(codebook_sizes.keys())
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(size) + 2, hint_dim)
                for name, size in codebook_sizes.items()
            }
        )
        self.proj = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hint_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in self.part_names
            }
        )
        self.mix = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, code_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = None
        count = 0
        ref = None
        for name in self.part_names:
            ids = code_inputs.get(name)
            if ids is None:
                continue
            ref = ids
            part_hidden = self.proj[name](self.embeddings[name](ids))
            hidden = part_hidden if hidden is None else hidden + part_hidden
            count += 1
        if hidden is None or count <= 0:
            if ref is None:
                raise ValueError("CodeHintEncoder requires at least one part tensor when called.")
            hidden = torch.zeros((*ref.shape, self.norm.normalized_shape[0]), device=ref.device, dtype=torch.float32)
            count = 1
        hidden = hidden / float(count)
        return self.norm(hidden + self.mix(hidden))


class SpatialConditionEncoder(nn.Module):
    def __init__(self, codebook_sizes: Dict[str, int], d_model: int, part_code_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.part_names = list(codebook_sizes.keys())
        self.part_embeddings = nn.ModuleDict({name: nn.Embedding(int(size), part_code_dim) for name, size in codebook_sizes.items()})
        self.missing_tokens = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(part_code_dim, dtype=torch.float32)) for name in self.part_names}
        )
        in_dim = len(self.part_names) * part_code_dim + len(self.part_names) * 2
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        *,
        code_targets: Dict[str, torch.Tensor],
        source_part_mask: torch.Tensor,
        target_part_mask: torch.Tensor,
    ) -> torch.Tensor:
        ref = next(iter(code_targets.values()))
        batch_size, seq_len = int(ref.shape[0]), int(ref.shape[1])
        feats = []
        for idx, name in enumerate(self.part_names):
            if name not in code_targets:
                raise KeyError(f"Missing code_targets entry for part '{name}'")
            emb = self.part_embeddings[name](code_targets[name].long())
            missing = self.missing_tokens[name].view(1, 1, -1).expand(batch_size, seq_len, -1).to(dtype=emb.dtype, device=emb.device)
            present = source_part_mask[:, idx].view(batch_size, 1, 1)
            feats.append(torch.where(present, emb, missing))

        presence = torch.cat([source_part_mask.float(), target_part_mask.float()], dim=-1)
        presence = presence.unsqueeze(1).expand(-1, seq_len, -1).to(dtype=feats[0].dtype, device=feats[0].device)
        hidden = self.input_proj(torch.cat([*feats, presence], dim=-1))
        return self.norm(hidden)


class AudioTextTokenPredictor(nn.Module):
    def __init__(
        self,
        *,
        prosody_dim: int,
        text_scalar_dim: int,
        vocab_size: int,
        codebook_sizes: Dict[str, int],
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_text: bool = True,
        use_code_hints: bool = False,
        code_hint_dim: int = 128,
        bert_text_dim: int = 0,
        spatial_code_dim: int = 128,
        num_spatial_tasks: int = 4,
    ):
        super().__init__()
        self.use_text = bool(use_text)
        self.use_code_hints = bool(use_code_hints)
        self.part_names = list(codebook_sizes.keys())
        self.rhythm_encoder = RhythmEncoder(prosody_dim, d_model, dropout=dropout)
        self.text_encoder = TextEncoder(vocab_size, text_scalar_dim, d_model, bert_dim=bert_text_dim, dropout=dropout)
        self.fusion = ContentRhythmFusion(d_model, dropout=dropout)
        self.code_hint_encoder = CodeHintEncoder(codebook_sizes, d_model, hint_dim=code_hint_dim, dropout=dropout)
        self.spatial_encoder = SpatialConditionEncoder(codebook_sizes, d_model, part_code_dim=spatial_code_dim, dropout=dropout)
        self.spatial_task_embedding = nn.Embedding(int(num_spatial_tasks), d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        self.code_hint_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.code_hint_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.code_hint_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, int(size)) for name, size in codebook_sizes.items()})

    def _run_temporal(self, hidden: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.positional_encoding(hidden)
        return self.temporal(hidden, src_key_padding_mask=key_padding_mask)

    def forward(
        self,
        *,
        token_prosody: torch.Tensor,
        token_word_ids: torch.Tensor,
        token_text_scalar: torch.Tensor,
        token_text_bert: Optional[torch.Tensor] = None,
        token_code_inputs: Optional[Dict[str, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        rhythm = self.rhythm_encoder(token_prosody)
        if self.use_text:
            content = self.text_encoder(token_word_ids, token_text_scalar, token_text_bert)
            fused, gate = self.fusion(rhythm, content)
        else:
            content = torch.zeros_like(rhythm)
            fused = rhythm
            gate = torch.zeros((*rhythm.shape[:2], 1), device=rhythm.device, dtype=rhythm.dtype)

        if self.use_code_hints and token_code_inputs is not None:
            code_hint = self.code_hint_encoder(token_code_inputs).to(dtype=fused.dtype)
            hint_gate = torch.sigmoid(self.code_hint_gate(torch.cat([fused, code_hint], dim=-1)))
            fused = self.code_hint_norm(fused + self.code_hint_out(hint_gate * code_hint))
        else:
            code_hint = torch.zeros_like(fused)
            hint_gate = torch.zeros((*fused.shape[:2], 1), device=fused.device, dtype=fused.dtype)

        fused = self._run_temporal(fused, key_padding_mask=key_padding_mask)
        logits = {name: head(fused) for name, head in self.heads.items()}
        return {
            "logits": logits,
            "fusion_gate": gate,
            "hint_gate": hint_gate,
            "rhythm_hidden": rhythm,
            "content_hidden": content,
            "code_hint_hidden": code_hint,
            "fused_hidden": fused,
        }

    def forward_spatial(
        self,
        *,
        code_targets: Dict[str, torch.Tensor],
        source_part_mask: torch.Tensor,
        target_part_mask: torch.Tensor,
        task_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        ref = next(iter(code_targets.values()))
        if task_ids.ndim == 0:
            task_ids = task_ids.view(1).expand(int(ref.shape[0]))
        spatial = self.spatial_encoder(
            code_targets=code_targets,
            source_part_mask=source_part_mask,
            target_part_mask=target_part_mask,
        )
        task_hidden = self.spatial_task_embedding(task_ids.long()).unsqueeze(1).expand(-1, int(ref.shape[1]), -1)
        fused = self._run_temporal(spatial + task_hidden.to(dtype=spatial.dtype), key_padding_mask=key_padding_mask)
        logits = {name: head(fused) for name, head in self.heads.items()}
        return {
            "logits": logits,
            "spatial_hidden": spatial,
            "task_hidden": task_hidden,
            "fused_hidden": fused,
        }
