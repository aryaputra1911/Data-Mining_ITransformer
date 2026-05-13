"""
Model architectures for GEMASTIK iTransformer Pipeline.
4 models: LSTM, PatchTST, Crossformer, iTransformer -- all self-contained.

v4 Deep Optimization:
  - All models accept (x_price, x_temporal) signature
  - iTransformer: Learnable temperature, RevIN (scale-only), TemporalEmbedding
  - RMSNorm, Gaussian noise, nn.Embedding for variate identity
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import N_VARIATES, N_TEMPORAL, LOOKBACK, HORIZON, N_LABELS, GAUSSIAN_NOISE_STD


# =============================================================================
# Model 1: LSTM Baseline (Weak Baseline)
# =============================================================================

class LSTMBaseline(nn.Module):
    """Downsized LSTM baseline. Gaussian noise + dropout 0.3."""

    def __init__(self, n_variates=N_VARIATES, hidden_size=64, num_layers=2,
                 dropout=0.3, horizon=HORIZON, n_labels=N_LABELS,
                 noise_std=GAUSSIAN_NOISE_STD):
        super().__init__()
        self.n_variates = n_variates
        self.horizon    = horizon
        self.noise_std  = noise_std

        self.lstm = nn.LSTM(
            input_size=n_variates, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            batch_first=True,
        )
        self.forecast_head = nn.Linear(hidden_size, horizon * n_variates)
        self.spike_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_labels),
        )

    def forward(self, x, x_temporal=None):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        out, (h, _) = self.lstm(x)
        last_hidden = h[-1]
        forecast = self.forecast_head(last_hidden)
        forecast = forecast.view(-1, self.horizon, self.n_variates)
        spike_logits = self.spike_head(last_hidden)
        return forecast, spike_logits


# =============================================================================
# Model 2: PatchTST (Strong Baseline)
# =============================================================================

class PatchEmbedding(nn.Module):
    """Split each variate's time series into patches and embed."""

    def __init__(self, patch_len, stride, d_model, lookback, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (lookback - patch_len) // stride + 1
        self.proj      = nn.Linear(patch_len, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(-1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens  = self.proj(patches)
        tokens  = tokens + self.pos_enc
        return self.dropout(tokens)


class PatchTSTWrapper(nn.Module):
    """PatchTST downsized: 1 layer, d=64, pooled spike head, Gaussian noise."""

    def __init__(self, n_variates=N_VARIATES, lookback=LOOKBACK, horizon=HORIZON,
                 patch_len=8, stride=4, d_model=64, n_heads=4,
                 n_layers=1, dropout=0.3, n_labels=N_LABELS,
                 noise_std=GAUSSIAN_NOISE_STD):
        super().__init__()
        self.n_variates = n_variates
        self.horizon    = horizon
        self.d_model    = d_model
        self.noise_std  = noise_std

        n_patches = (lookback - patch_len) // stride + 1
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, lookback, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-variate forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(n_patches * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )
        # Pooled spike head (avg over variates, not concat) to reduce params
        self.spike_head = nn.Sequential(
            nn.Linear(n_patches * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_labels),
        )
        self.n_patches = n_patches

    def forward(self, x, x_temporal=None):
        B = x.size(0)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        forecasts = []
        all_encoded = []

        for v in range(self.n_variates):
            x_v     = x[:, :, v].unsqueeze(-1)
            patches = self.patch_embed(x_v)
            encoded = self.encoder(patches)
            flat    = encoded.reshape(B, -1)          # (B, n_patches*d_model)
            fc_v    = self.forecast_head(flat)
            forecasts.append(fc_v)
            all_encoded.append(flat)

        forecast = torch.stack(forecasts, dim=2)
        # Pool over variates instead of concat (reduces params massively)
        stacked = torch.stack(all_encoded, dim=1)     # (B, V, n_patches*d)
        pooled  = stacked.mean(dim=1)                  # (B, n_patches*d)
        spike_logits = self.spike_head(pooled)

        return forecast, spike_logits


# =============================================================================
# Model 3: Crossformer (Direct Competitor)
# =============================================================================

class TwoStageAttention(nn.Module):
    """Two-Stage Attention (TSA): Crossformer's core mechanism."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.time_ff_norm = nn.LayerNorm(d_model)

        self.var_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.var_norm  = nn.LayerNorm(d_model)
        self.var_ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.var_ff_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B_V, S, D = x.shape
        residual = x
        x_norm = self.time_norm(x)
        x_attn, _ = self.time_attn(x_norm, x_norm, x_norm)
        x = residual + x_attn
        residual = x
        x = residual + self.time_ff(self.time_ff_norm(x))
        return x


class CrossformerWrapper(nn.Module):
    """Crossformer downsized: 1 layer, d=64, pooled spike head, Gaussian noise."""

    def __init__(self, n_variates=N_VARIATES, lookback=LOOKBACK, horizon=HORIZON,
                 seg_len=6, d_model=64, n_heads=4, n_layers=1,
                 dropout=0.3, n_labels=N_LABELS,
                 noise_std=GAUSSIAN_NOISE_STD):
        super().__init__()
        self.n_variates = n_variates
        self.horizon    = horizon
        self.seg_len    = seg_len
        self.d_model    = d_model
        self.noise_std  = noise_std

        self.n_segments = max(1, lookback // seg_len)
        self.seg_embed = nn.Linear(seg_len, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, self.n_segments, d_model) * 0.02)

        self.layers = nn.ModuleList([
            TwoStageAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.var_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.var_norm = nn.LayerNorm(d_model)

        self.forecast_head = nn.Sequential(
            nn.Linear(self.n_segments * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )
        # Pooled spike head
        self.spike_head = nn.Sequential(
            nn.Linear(self.n_segments * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_labels),
        )

    def forward(self, x, x_temporal=None):
        B = x.size(0)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        x_trunc = x[:, :self.n_segments * self.seg_len, :]
        x_seg = x_trunc.reshape(B, self.n_segments, self.seg_len, self.n_variates)

        var_reprs = []
        for v in range(self.n_variates):
            seg_v   = x_seg[:, :, :, v]
            tokens  = self.seg_embed(seg_v) + self.pos_enc
            for layer in self.layers:
                tokens = layer(tokens)
            var_reprs.append(tokens)

        var_stack = torch.stack(var_reprs, dim=1)
        B_, V_, S_, D_ = var_stack.shape
        var_mix = var_stack.permute(0, 2, 1, 3).reshape(B_ * S_, V_, D_)
        var_mix_norm = self.var_norm(var_mix)
        var_mix_attn, _ = self.var_attn(var_mix_norm, var_mix_norm, var_mix_norm)
        var_mix = var_mix + var_mix_attn
        var_mix = var_mix.reshape(B_, S_, V_, D_).permute(0, 2, 1, 3)

        forecasts = []
        all_flat  = []
        for v in range(self.n_variates):
            flat = var_mix[:, v].reshape(B, -1)
            fc_v = self.forecast_head(flat)
            forecasts.append(fc_v)
            all_flat.append(flat)

        forecast = torch.stack(forecasts, dim=2)
        # Pool over variates for spike head
        stacked = torch.stack(all_flat, dim=1)     # (B, V, n_seg*d)
        pooled  = stacked.mean(dim=1)               # (B, n_seg*d)
        spike_logits = self.spike_head(pooled)

        return forecast, spike_logits


# =============================================================================
# iTransformer building blocks
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).
    v4: Scale-only mode (no mean centering) for log-return data that is
    already centered around zero. Only normalizes by std."""

    def __init__(self, n_variates, eps=1e-5, affine=True):
        super().__init__()
        self.n_variates = n_variates
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_variates))
            self.beta  = nn.Parameter(torch.zeros(1, 1, n_variates))
        self._std = None

    def normalize(self, x):
        """x: (B, T, V) -- scale-only normalization (no mean shift)."""
        self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x_norm = x / self._std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm

    def denormalize(self, x):
        """x: (B, T, V) -- reverse the scaling."""
        if self.affine:
            x = (x - self.beta) / self.gamma
        return x * self._std


class TemporalEmbedding(nn.Module):
    """Project 15 temporal/context features into d_model space.
    Averages over the lookback window to create a global context vector
    that is added to every variate token."""

    def __init__(self, n_temporal, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_temporal, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x_temporal):
        """
        x_temporal: (B, T, n_temporal)
        Returns: (B, 1, d_model) -- global context vector
        """
        # Average over time dimension, then project
        context = x_temporal.mean(dim=1, keepdim=True)        # (B, 1, n_temporal)
        return self.proj(context)                              # (B, 1, d_model)


class TemperatureAttentionLayer(nn.Module):
    """Transformer encoder layer with LEARNABLE temperature-scaled attention.
    v4: tau is nn.Parameter initialized to 0.5, learned during training.
    Softmax(QK^T / (tau * sqrt(d_k))) -- tau < 1 sharpens attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.2, init_temperature=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        # v4: Learnable temperature parameter
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_temperature)))

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # RMSNorm (Pre-norm architecture)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Storage for attention analysis
        self._attn_weights = None

    @property
    def temperature(self):
        """Ensure temperature stays positive via exp(log_tau)."""
        return self.log_tau.exp()

    def _scaled_dot_product_attention(self, q, k, v, need_weights=False):
        """Attention: softmax(QK^T / (tau * sqrt(d_k)))."""
        scale = (self.head_dim ** 0.5) * self.temperature
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if need_weights:
            self._attn_weights = attn_weights.detach()
        return output

    def forward(self, x, need_weights=False):
        B, N, D = x.shape

        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = self._scaled_dot_product_attention(q, k, v, need_weights)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out

        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# Model 4: iTransformer (Proposed) -- v4 Deep Optimization
# =============================================================================

class iTransformerModel(nn.Module):
    """
    iTransformer v4 -- Inverted attention with deep optimization.

    Enhancements:
      1. RMSNorm replaces LayerNorm
      2. LEARNABLE temperature in attention (init=0.5)
      3. RevIN scale-only (no mean shift for centered log-returns)
      4. TemporalEmbedding: 15 context features -> d_model context vector
      5. nn.Embedding for variate identity
      6. Gaussian noise injection during training
      7. Compact: d_model=64, n_heads=4, n_layers=2

    Input:
      x_price:    (B, 30, 33)  -- price log-returns
      x_temporal: (B, 30, 15)  -- calendar/event/spread context
    Output:
      forecast:     (B, 14, 33)
      spike_logits: (B, 30)
    """

    def __init__(self, n_variates=N_VARIATES, n_temporal=N_TEMPORAL,
                 lookback=LOOKBACK, horizon=HORIZON,
                 d_model=None, n_heads=None, n_layers=None,
                 d_ff=None, dropout=None, n_labels=N_LABELS,
                 temperature=None, noise_std=None):
        super().__init__()
        from config import (ITF_D_MODEL, ITF_N_HEADS, ITF_N_LAYERS, ITF_D_FF,
                            ITF_DROPOUT, ITF_ATTN_TEMP, GAUSSIAN_NOISE_STD)

        d_model     = d_model or ITF_D_MODEL
        n_heads     = n_heads or ITF_N_HEADS
        n_layers    = n_layers or ITF_N_LAYERS
        d_ff        = d_ff or ITF_D_FF
        dropout     = dropout if dropout is not None else ITF_DROPOUT
        temperature = temperature or ITF_ATTN_TEMP
        noise_std   = noise_std if noise_std is not None else GAUSSIAN_NOISE_STD

        self.n_variates = n_variates
        self.n_temporal = n_temporal
        self.lookback   = lookback
        self.horizon    = horizon
        self.d_model    = d_model
        self.noise_std  = noise_std

        # RevIN -- scale-only for centered log-returns
        self.revin = RevIN(n_variates, affine=True)

        # Temporal projection: project each variate's time series to d_model
        self.temporal_proj = nn.Linear(lookback, d_model)

        # Learnable variate identity embedding
        self.variate_embed = nn.Embedding(n_variates, d_model)

        # v4: TemporalEmbedding for 15 context features
        self.temporal_embedding = TemporalEmbedding(n_temporal, d_model, dropout)

        # Temperature-scaled attention encoder layers (learnable tau)
        self.encoder_layers = nn.ModuleList([
            TemperatureAttentionLayer(d_model, n_heads, d_ff, dropout, temperature)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Forecast head
        self.forecast_head = nn.Linear(d_model, horizon)

        # CLS token for spike classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.spike_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_labels),
        )

    def forward(self, x, x_temporal=None, return_attention=False):
        """
        Args:
            x:          (B, lookback, n_variates) -- price log-returns
            x_temporal: (B, lookback, n_temporal) -- context features (optional)
        """
        B = x.size(0)

        # Gaussian noise injection during training
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # RevIN normalize (scale-only)
        x = self.revin.normalize(x)

        # Transpose: variates become tokens
        x_t = x.transpose(1, 2)                                    # (B, V, T)

        # Temporal projection
        variate_tokens = self.temporal_proj(x_t)                    # (B, V, d_model)

        # Add variate identity embeddings
        var_ids = torch.arange(self.n_variates, device=x.device)
        variate_tokens = variate_tokens + self.variate_embed(var_ids)

        # v4: Add temporal context embedding (broadcast to all variate tokens)
        if x_temporal is not None:
            temporal_ctx = self.temporal_embedding(x_temporal)       # (B, 1, d_model)
            variate_tokens = variate_tokens + temporal_ctx           # broadcast

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, variate_tokens], dim=1)            # (B, V+1, d_model)

        # Temperature-scaled encoder layers
        for layer in self.encoder_layers:
            tokens = layer(tokens, need_weights=return_attention)

        encoded = self.final_norm(tokens)

        # Forecast head (skip CLS)
        variate_encoded = encoded[:, 1:, :]
        forecast = self.forecast_head(variate_encoded)              # (B, V, horizon)
        forecast = forecast.transpose(1, 2)                         # (B, horizon, V)

        # RevIN denormalize forecast
        forecast = self.revin.denormalize(forecast)

        # Spike classification (CLS token)
        cls_encoded  = encoded[:, 0, :]
        spike_logits = self.spike_head(cls_encoded)

        return forecast, spike_logits

    def get_attention_matrix(self, x, x_temporal=None):
        """Extract attention matrix for propagation analysis."""
        self.eval()
        with torch.no_grad():
            B = x.size(0)

            x_normed = self.revin.normalize(x)
            x_t = x_normed.transpose(1, 2)
            tokens = self.temporal_proj(x_t)
            var_ids = torch.arange(self.n_variates, device=x.device)
            tokens = tokens + self.variate_embed(var_ids)

            if x_temporal is not None:
                temporal_ctx = self.temporal_embedding(x_temporal)
                tokens = tokens + temporal_ctx

            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            attn_matrices = []
            for layer in self.encoder_layers:
                tokens = layer(tokens, need_weights=True)
                if layer._attn_weights is not None:
                    attn = layer._attn_weights.mean(dim=1)
                    attn_matrices.append(attn)

            if attn_matrices:
                avg_attn = torch.stack(attn_matrices).mean(dim=0)
                avg_attn = avg_attn.mean(dim=0)
                variate_attn = avg_attn[1:, 1:]
            else:
                variate_attn = torch.zeros(self.n_variates, self.n_variates)

        return variate_attn.cpu().numpy()

    def get_learned_temperature(self):
        """Return current learned temperature values for all layers."""
        temps = []
        for i, layer in enumerate(self.encoder_layers):
            temps.append((i, layer.temperature.item()))
        return temps


# =============================================================================
# Factory functions
# =============================================================================

def build_all_models():
    """Instantiate all 4 models — baselines downsized for fair comparison."""
    from config import (ITF_D_MODEL, ITF_N_HEADS, ITF_N_LAYERS,
                        ITF_D_FF, ITF_DROPOUT, ITF_ATTN_TEMP,
                        BASE_D_MODEL, BASE_N_HEADS, BASE_N_LAYERS, BASE_DROPOUT)
    models = {
        "LSTM": LSTMBaseline(
            n_variates=N_VARIATES, hidden_size=64, num_layers=2,
            dropout=BASE_DROPOUT, horizon=HORIZON, n_labels=N_LABELS,
        ),
        "PatchTST": PatchTSTWrapper(
            n_variates=N_VARIATES, lookback=LOOKBACK, horizon=HORIZON,
            patch_len=8, stride=4, d_model=BASE_D_MODEL, n_heads=BASE_N_HEADS,
            n_layers=BASE_N_LAYERS, dropout=BASE_DROPOUT, n_labels=N_LABELS,
        ),
        "Crossformer": CrossformerWrapper(
            n_variates=N_VARIATES, lookback=LOOKBACK, horizon=HORIZON,
            seg_len=6, d_model=BASE_D_MODEL, n_heads=BASE_N_HEADS,
            n_layers=BASE_N_LAYERS, dropout=BASE_DROPOUT, n_labels=N_LABELS,
        ),
        "iTransformer": iTransformerModel(
            n_variates=N_VARIATES, n_temporal=N_TEMPORAL,
            lookback=LOOKBACK, horizon=HORIZON,
            d_model=ITF_D_MODEL, n_heads=ITF_N_HEADS, n_layers=ITF_N_LAYERS,
            d_ff=ITF_D_FF, dropout=ITF_DROPOUT, n_labels=N_LABELS,
            temperature=ITF_ATTN_TEMP,
        ),
    }
    return models


def build_ablation_models():
    """Instantiate iTransformer ablation variants."""
    from config import ITF_ATTN_TEMP
    configs = {
        "iTransformer_full": dict(
            n_layers=2, d_model=64, n_heads=4, d_ff=128,
            dropout=0.2, temperature=ITF_ATTN_TEMP,
        ),
        "iTransformer_1layer": dict(
            n_layers=1, d_model=64, n_heads=4, d_ff=128,
            dropout=0.2, temperature=ITF_ATTN_TEMP,
        ),
        "iTransformer_no_temp": dict(
            n_layers=2, d_model=64, n_heads=4, d_ff=128,
            dropout=0.2, temperature=1.0,
        ),
        "iTransformer_no_context": dict(
            n_layers=2, d_model=64, n_heads=4, d_ff=128,
            dropout=0.2, temperature=ITF_ATTN_TEMP, n_temporal=0,
        ),
    }
    models = {}
    for name, cfg in configs.items():
        n_temp = cfg.pop('n_temporal', N_TEMPORAL)
        models[name] = iTransformerModel(
            n_variates=N_VARIATES, n_temporal=n_temp,
            lookback=LOOKBACK, horizon=HORIZON,
            n_labels=N_LABELS, **cfg,
        )
    return models
