''' Regression models for meta-learning (MAML / ANIL).'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Regressor(nn.Module):
    ''' Simple regression model to start with.'''
    def __init__(self, in_dim:int=38, hidden_dims: tuple[int]=(40), out_dim:int =38):
        """Fully connected regression network. See the arguments"""
        super().__init__()
        layers = []
        last_h = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_h, h), nn.ReLU()]
            last_h = h
        layers += [nn.Linear(last_h, out_dim)]
        self.net = nn.Sequential(*layers)
        self.features = nn.Sequential(*layers[:-1]) # body
        self.head = layers[-1] # head

    def forward(self, x):
        """Forward pass through the full network
        x: [B, in_dim]
        return: [B, out_dim]"""
        return self.net(x)

class LSTMBody(nn.Module):
    """LSTM body — frozen in ANIL inner loop.
    
    LSTM Encoder-Decoder body — frozen in ANIL inner loop.

    Encoder : processes look_back steps → hidden state (h, c)
    Decoder : uses (h, c) + zero inputs → produces look_ahead steps
    Zero decoder inputs = "nothing known about future beyond encoded context"
    This is standard seq2seq for offline forecasting.

    Shape: [B, look_back, input_size] → [B, look_ahead, hidden_size]
    """
    def __init__(self, encoder: nn.LSTM, decoder:nn.LSTM, look_ahead: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.look_ahead = look_ahead

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full look_back window → compress to hidden state
        x: [B, look_back, input_size] -> Argument
        decoder output: [B, look_ahead, hidden_size]
        """
        _, (h, c) = self.encoder(x)                          # h,c: [layers, B, H]

        # Decode: zero inputs — hidden state carries all context
        dec_input = torch.zeros(
            x.size(0), self.look_ahead, self.decoder.input_size,
            device=x.device, dtype=x.dtype,
        )                                                     # [B, look_ahead, 1]
        out, _ = self.decoder(dec_input, (h, c))             # [B, look_ahead, H]
        return out

class LSTMHead(nn.Module):
    """Linear head — adapted in ANIL inner loop."""
    def __init__(self, fc: nn.Linear):
        super().__init__()
        self.fc = fc
        nn.init.xavier_uniform_(fc.weight)
        nn.init.zeros_(fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden features to output
        x: [B, look_ahead, hidden_size]

        Normalized (0,1) --> same as the target/output
        returns: Predictions [B, look_ahead, output_size]"""
        return torch.relu(self.fc(x))      # [B, look_ahead, output_size]


class LSTMRegressor(nn.Module):
    """
    LSTM Encoder-Decoder for time series forecasting.
    Compatible with MAML/ANIL (features + head attributes).

    Input  : [B, look_back,  input_size]
    Output : [B, look_ahead, output_size]
    """
    def __init__(
        self,
        input_size:  int   = 3,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        output_size: int   = 1,
        look_ahead:  int   = 96,
        dropout:     float = 0.2,
    ):
        # Inherit everything from the nn.Module (parent class)
        super().__init__()
        self.look_ahead  = look_ahead
        self.output_size = output_size

        # ── Encoder ──────────────────────────────────────────────
        # Plays the role of "features" extractor (like Regressor.features)
        encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        # ── Decoder — generates future steps from hidden state ────
        # input_size=1: zero placeholder (no future inputs in LSTM mode)
        decoder = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        # Linear(hidden_size → output_size): LSTMBody returns [B, look_ahead, H]
        self.fc  = nn.Linear(hidden_size, output_size)
        # self.drop = nn.Dropout(dropout)

        # ANIL split — body frozen, head adapted
        self.features = LSTMBody(encoder, decoder, look_ahead)
        self.head     = LSTMHead(self.fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: encode look_back → decode look_ahead → project"""
        return self.head(self.features(x)) # [B, look_ahead, output_size]

# ==================================================================
# TFT IMPLEMENTATION (FEATURES OR BODY + HEAD)
# ==================================================================

class TFTBody(nn.Module):
    """TFT feature extractor — frozen in ANIL inner loop."""
    def __init__(self, enc_size, dec_size, hidden_size,
                 num_heads, num_layers, dropout):
        super().__init__()
        self.vsn_enc      = VariableSelectionNetwork(enc_size, hidden_size, dropout)
        self.vsn_dec      = VariableSelectionNetwork(dec_size, hidden_size, dropout)
        self.enc_proj     = nn.Linear(enc_size, hidden_size)
        self.dec_proj     = nn.Linear(dec_size, hidden_size)
        self.drop         = nn.Dropout(dropout)
        self.encoder      = nn.LSTM(hidden_size, hidden_size,
                                    num_layers=num_layers, batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
        self.encoder_grn  = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.attn         = nn.MultiheadAttention(hidden_size, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.attn_norm    = nn.LayerNorm(hidden_size)
        self.post_attn_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)

    def forward(self, x_enc, x_dec):
        """
        Run TFT body.
        Refer to Lim et al. (2021) for architectural details."""
        # Variable selection on encoder
        x_enc = self.vsn_enc(x_enc)
        h_enc = self.drop(self.enc_proj(x_enc))
        # Temporal processing via LSTM
        enc, _ = self.encoder(h_enc)
        enc = self.encoder_grn(enc)
        # Decoder queries
        x_dec = self.vsn_dec(x_dec)
        q = self.drop(self.dec_proj(x_dec))
        attn_out, _ = self.attn(q, enc, enc)
        attn_out = self.attn_norm(attn_out + q)
        return self.post_attn_grn(attn_out)          # [B, look_ahead, H]

class TFTHead(nn.Module):
    """Expanded head for ANIL — more capacity for fast adaptation."""
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # One cannot pass softplus, because we need to have zero values at night!!
        # return F.softplus(self.head(x))   # smooth, always-positive, no dead zone

        # Don't use sigmoid here — let the loss handle quantile crossings
        # Consider using RELU
        return torch.relu(self.head(x))  # [B, look_ahead, output_size]

class TFTRegressor(nn.Module):
    """
    Lightweight Temporal Fusion Transformer compatible with MAML inner-loop.

    Architecture:
        1. Variable Selection Network   — soft feature importance
        2. Input projection             — F → H
        3. LSTM encoder                 — temporal processing of look-back
        4. Encoder GRN                  — gated residual on encoder output
        5. Learned decoder queries      — one embedding per forecast step
        6. Interpretable MHA            — cross-attn: future queries × past enc
        7. Post-attention GRN           — final gating before output
        8. Output projection            — H → output_size

    I/O contract (identical to LSTMRegressor):
        x   : [B, look_back,  n_features]
        out : [B, look_ahead, output_size]

    MAML: forward(x_enc, x_dec) adapts all parameters
    ANIL: features_forward(x_enc, x_dec) frozen; head (output_proj) adapts only
    """
    def __init__(self, config: dict):
        super().__init__()

        # ── Read enc/dec sizes (replaces old input_size) ──────────────
        enc_size    = config["enc_size"]        # encoder input features
        dec_size    = config["dec_size"]        # decoder input features
        hidden_size = config["hidden_size"]
        num_heads   = config.get("num_heads",    4)
        num_layers  = config.get("num_layers",   2)
        dropout     = config.get("dropout",      0.1)
        output_size = config.get("output_size",  1)
        self.look_ahead = config["look_ahead"]

        # ── ANIL-ready split ──────────────────────────────────────────
        self.features = TFTBody(enc_size, dec_size, hidden_size,
                                num_heads, num_layers, dropout)
        self.head     = TFTHead(hidden_size, output_size)

    def features_forward(self, x_enc, x_dec) -> torch.Tensor:
        """Backward compat — delegates to TFTBody."""
        return self.features(x_enc, x_dec)

    def forward(self, x_enc, x_dec) -> torch.Tensor:
        # ✅ MAML unchanged
        return self.head(self.features(x_enc, x_dec))   

class GRN(nn.Module):
    """
    Gated Residual Network (Lim et al. 2021, Eq. 3-4).

    η2 = ELU(W1·x + b1)
    gate = σ(W_gate·η2)
    out  = LayerNorm(gate ⊙ W2·η2 + skip(x))
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta  = F.elu(self.fc1(x))
        eta2 = self.dropout(self.fc2(eta))
        g    = torch.sigmoid(self.gate(eta))
        out  = g * eta2 + self.skip(x)

        return self.norm(out)


class VariableSelectionNetwork(nn.Module):
    """
    Soft feature selection via per-timestep GRN + softmax weighting.
    Learns which input variables matter at each step.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.grn = GRN(input_size, hidden_size, input_size, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-variable importance weights and apply element-wise"""
        grn_out = self.grn(x)

        # TO BE REMOVED --> NOT ANYMORE IS ZERO 
        # Guard: NaN in GRN poisons softmax → entire feature tensor
        if torch.isnan(grn_out).any():
            grn_out = torch.zeros_like(grn_out)   # uniform weights fallback

        weights = self.softmax(grn_out)   # [B, T, F]
        return x * weights

# ════════════════════════════════════════════════════════════════════════════
# TCN COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

class TCNResidualBlock(nn.Module):
    """
    Non-causal dilated residual block.

    Design choices for MAML stability:
      • Symmetric padding : non-causal, attends past AND future
                            valid for offline PV forecasting
      • LayerNorm         : works with ANY batch size (unlike BatchNorm)
                            critical for small MAML inner batches
      • GELU              : smooth gradients vs ReLU — better inner-loop convergence
      • Residual skip     : prevents vanishing meta-gradients

    kernel_size MUST be odd → exact symmetric padding:
        pad = dilation × (kernel_size - 1) // 2  (each side)

    Shape: [B, n_filters, T] → [B, n_filters, T]
    """
    def __init__(
        self,
        n_filters:   int,
        kernel_size: int,
        dilation:    int,
        dropout:     float,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, \
            "kernel_size must be odd for exact symmetric non-causal padding."
        pad = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            n_filters, n_filters, kernel_size,
            dilation=dilation, padding=pad,
        )
        self.conv2 = nn.Conv1d(
            n_filters, n_filters, kernel_size,
            dilation=dilation, padding=pad,
        )
        # LayerNorm over channel dim — applied via transpose trick:
        #   [B, C, T] → [B, T, C] → LayerNorm(C) → [B, C, T]
        self.norm1   = nn.LayerNorm(n_filters)
        self.norm2   = nn.LayerNorm(n_filters)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x

        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)

        return out + residual   # residual connection


class TCNBody(nn.Module):
    """
    TCN feature extractor.

    Role in meta-learning:
      MAML : entire model adapted in inner loop
      ANIL : this body is FROZEN in inner loop (future upgrade)

    Architecture:
      input_proj : Conv1d(input_size → n_filters, kernel=1)  — feature mixing
      blocks     : n_blocks × TCNResidualBlock, dilation = 2^i

    Receptive field (non-causal, kernel=3, n_blocks=6):
      RF = 1 + 2×(3-1)×(1+2+4+8+16+32) = 253 > look_back=96  ✓

    Shape: [B, look_back, input_size] → [B, n_filters, look_back]
    """

    def __init__(
        self,
        input_size:  int,
        n_filters:   int,
        kernel_size: int,
        n_blocks:    int,
        dropout:     float,
    ) -> None:
        super().__init__()
        # 1×1 conv to project input features into filter space
        self.input_proj = nn.Conv1d(input_size, n_filters, kernel_size=1)

        # Exponentially growing dilation: 1, 2, 4, 8, 16, 32, ...
        self.blocks = nn.ModuleList([
            TCNResidualBlock(
                n_filters   = n_filters,
                kernel_size = kernel_size,
                dilation    = 2 ** i,
                dropout     = dropout,
            )
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]  (time-first from PVTimeSeriesDatasetV2 LSTM mode)
        out = x.permute(0, 2, 1)    # → [B, C, T]  (channel-first for Conv1d)
        out = self.input_proj(out)  # → [B, n_filters, T]
        for block in self.blocks:
            out = block(out)        # → [B, n_filters, T]
        return out                  # [B, n_filters, look_back]


class TCNHead(nn.Module):
    """
    TCN output head — adapted per task in ANIL; full model adapted in MAML.

    Two-step projection preserves temporal structure of the forecast:
      temporal_proj : Linear(look_back  → look_ahead)
                      each output step = learned weighted mix of all input steps
      channel_proj  : Linear(n_filters  → output_size)
                      regression to target

    Shape: [B, n_filters, look_back] → [B, look_ahead, output_size]

    Parameter count (n_filters=64, look_back=look_ahead=96, output_size=1):
      temporal_proj : 96×96 + 96 = 9,312 params
      channel_proj  : 64× 1 + 1  =    65 params
      Total head    :               9,377 params  → fast ANIL adaptation
    """

    def __init__(
        self,
        n_filters:   int,
        # look_back:   int,
        look_ahead:  int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.look_ahead   = look_ahead
        self.channel_proj = nn.Linear(n_filters, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_filters, look_back]  — arbitrary look_back
        out = F.interpolate(
            x, size=self.look_ahead,
            mode='linear', align_corners=False,
        )                               # [B, n_filters, look_ahead]
        out = out.permute(0, 2, 1)      # [B, look_ahead, n_filters]
        return torch.sigmoid(self.channel_proj(out))   # [B, look_ahead, output_size]


class TCNRegressor(nn.Module):
    """
    Non-causal TCN for PV power point forecasting.

    ── Interface ────────────────────────────────────────────────────────────
    Identical to LSTMRegressor → ZERO changes to LightningMAMLModule,
    PVTimeSeriesDatasetV2, or any sampler/loss wiring.

        Input : [B, look_back,  input_size]   (LSTM-mode dataset)
        Output: [B, look_ahead, output_size]  (point forecast, output_size=1)

    ── ANIL readiness ───────────────────────────────────────────────────────
    self.features = TCNBody  → frozen in ANIL inner loop (future)
    self.head     = TCNHead  → adapted in ANIL inner loop (future)
    LightningANILModule already reads self.meta_model.features / .head  ✓
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        input_size  = config.get("input_size",  13)
        n_filters   = config.get("n_filters",   64)
        kernel_size = config.get("kernel_size",  3)
        n_blocks    = config.get("n_blocks",     6)
        dropout     = config.get("dropout",    0.1)
        look_back   = config.get("look_back",   96)
        look_ahead  = config.get("look_ahead",  96)
        output_size = config.get("output_size",  1)

        self.features = TCNBody(
            input_size  = input_size,
            n_filters   = n_filters,
            kernel_size = kernel_size,
            n_blocks    = n_blocks,
            dropout     = dropout,
        )
        self.head = TCNHead(
            n_filters   = n_filters,
            # look_back   = look_back,
            look_ahead  = look_ahead,
            output_size = output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, look_back, input_size]
        return self.head(self.features(x))  # [B, look_ahead, output_size]


# ════════════════════════════════════════════════════════════════════════════
# PYTORCH FORECASTING TFT WRAPPER
# ════════════════════════════════════════════════════════════════════════════

class PlainTFT(nn.Module):
    """
    Thin wrapper around pytorch_forecasting's TemporalFusionTransformer.

    Purpose:
        - Makes the TFT look like a plain nn.Module for the meta-learning pipeline
        - forward(x_dict) → prediction tensor (not dict)
        - Provides .features and .head attributes for ANIL split

    ANIL split (default — minimal head):
        - head    = output_layer only (Linear or ModuleList of Linears)
        - features = everything else (embeddings, VSN, LSTMs, attention, GRNs)

    I/O contract:
        Input  : x_dict (pytorch_forecasting format from PVTimeSeriesDatasetV2._to_pf_dict)
        Output : prediction tensor [B, look_ahead, output_size]
    """

    def __init__(self, tft):
        """
        Args:
            tft: TemporalFusionTransformer instance from from_dataset()
        """
        super().__init__()
        self.tft = tft # TFT is the nn.Module
        # # everything except the output_layer
        # self._features = _TFTFeatures(tft)

        # # ── ANIL interface ────────────────────────────────────────
        # # head = only the output_layer (adapted per-task in inner loop)
        # # features = everything else (frozen in inner loop)
        # self._head = tft.output_layer

        
        # ── Expand head: replace single Linear with MLP ───────────
        hidden_size = tft.hparams.hidden_size  # 64
        output_size = tft.hparams.output_size  # 7

        new_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(), # probablilistic approach to the input weights (filter)
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        # Replace inside TFT so full-forward path also uses it
        tft.output_layer = new_head

        self._features = _TFTFeatures(tft)
        self._head = new_head

    @property
    def features(self):
        """Return the feature extractor (everything except output_layer).
        Used by LightningANILModule.init_algorithm()."""
        return self._features

    @property
    def head(self):
        """Return the head (output_layer only).
        Used by LightningANILModule.init_algorithm()."""
        return self._head

    def forward(self, x):
        """
        Args:
            x: dict with pytorch_forecasting TFT keys
               OR Tensor [B, look_ahead, hidden_size] (from ANIL features encoding)

        Returns:
            prediction tensor [B, look_ahead, output_size]
        """
        if isinstance(x, dict):
            device = next(self.tft.parameters()).device
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in x.items()}
            # Force nested LightningModule's .device property
            self.tft._device = device
            out = self.tft(x) # Called directly here
            prediction = out["prediction"] # extract prediction tensor
            if isinstance(prediction, (list, tuple)):
                prediction = torch.stack(prediction, dim=-1)
            return prediction # plain tensor, not dict
        else:
            return self._head(x)
        
    def get_interpretation(self):
        """Return the last stored interpretation dict from features forward pass."""
        return self._features.last_interpretation

class _TFTFeatures(nn.Module):
    """
    TFT feature extractor for ANIL — everything EXCEPT output_layer.

    This module runs the full TFT forward MINUS the final linear projection.
    Returns the pre-output representation [B, look_ahead, hidden_size].

    NOTE: This is a view into the same parameters as PlainTFT.tft —
          no parameter duplication.
    """

    def __init__(self, tft):
        super().__init__()
        self.tft = tft
        self.last_interpretation = None # stores attention + variable weights

    def forward(self, x):
        """
        Run TFT up to (but not including) output_layer.

        Args:
            x: dict with pytorch_forecasting TFT keys

        Returns:
            pre_output: [B, look_ahead, hidden_size]
        """
        # Use TFT's internal method to get intermediate representation
        # The TFT forward does: features → output_layer → transform_output
        # We want just the features part.
        #
        # Strategy: run full forward, intercept before output_layer
        # by temporarily replacing output_layer with identity

        device = next(self.tft.parameters()).device
        if isinstance(x, dict):
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in x.items()}

        # Force nested LightningModule's .device property to return correct device
        # (Trainer only sets _device on top-level module, not nested submodules)
        self.tft._device = device

        original_output_layer = self.tft.output_layer
        identity_layer = nn.Identity().to(device)
        self.tft.output_layer = identity_layer

        try:
            out = self.tft(x)
            pre_output = out["prediction"]

            # Store interpretation data (attention + variable selection weights)
            self.last_interpretation = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in out.items()
                if k != "prediction"
            }
        finally:
            self.tft.output_layer = original_output_layer

        return pre_output