"""
TFT Interpretation plots using pytorch_forecasting's built-in outputs.

Generates:
    1. Encoder variable importance (which input features matter in look_back)
    2. Decoder variable importance (which input features matter in look_ahead)
    3. Attention heatmap (which past timesteps influence each prediction step)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

class TFTInterpretation:
    """
    Collects and plots TFT interpretation data accumulated during meta-testing.

    Usage:
        interp = TFTInterpretation(save_path="output/plots/", tft_model=plain_tft)
        # During testing, call accumulate() after each task
        interp.accumulate(plain_tft.get_interpretation())
        # After all tasks:
        interp.plot_all()
    """

    def __init__(
        self,
        save_path: str,
        encoder_variables: List[str],
        decoder_variables: List[str],
        static_variables: List[str] = None,
    ):
        self.save_path = save_path.rstrip("/") + "/"
        os.makedirs(self.save_path, exist_ok=True)

        self.encoder_variables = encoder_variables
        self.decoder_variables = decoder_variables
        self.static_variables = static_variables or []

        # Accumulators
        self.encoder_weights_list = []
        self.decoder_weights_list = []
        self.static_weights_list = []
        self.attention_list = []

    def accumulate(self, interpretation: Optional[Dict]):
        """
        Accumulate one task's interpretation data.

        Args:
            interpretation: dict from PlainTFT.get_interpretation()
                Keys: encoder_attention, decoder_attention,
                      encoder_variables, decoder_variables, static_variables
        """
        if interpretation is None:
            return

        if "encoder_variables" in interpretation:
            w = interpretation["encoder_variables"]
            if isinstance(w, torch.Tensor) and w.numel() > 0:
                # [B, T, n_variables, 1] or [B, n_variables] depending on version
                if w.dim() == 4:
                    w = w.squeeze(-1).mean(dim=(0, 1))  # avg over batch + time
                elif w.dim() == 3:
                    w = w.squeeze(-1).mean(dim=0)  # avg over batch
                elif w.dim() == 2:
                    w = w.mean(dim=0)
                self.encoder_weights_list.append(w.numpy())

        if "decoder_variables" in interpretation:
            w = interpretation["decoder_variables"]
            if isinstance(w, torch.Tensor) and w.numel() > 0:
                if w.dim() == 4:
                    w = w.squeeze(-1).mean(dim=(0, 1))
                elif w.dim() == 3:
                    w = w.squeeze(-1).mean(dim=0)
                elif w.dim() == 2:
                    w = w.mean(dim=0)
                self.decoder_weights_list.append(w.numpy())

        if "static_variables" in interpretation:
            w = interpretation["static_variables"]
            if isinstance(w, torch.Tensor) and w.numel() > 0:
                if w.dim() >= 2:
                    w = w.mean(dim=0)
                self.static_weights_list.append(w.numpy())

        if "encoder_attention" in interpretation:
            attn = interpretation["encoder_attention"]
            if isinstance(attn, torch.Tensor) and attn.numel() > 0:
                # [B, n_heads, decoder_len, encoder_len]
                # Average over batch and heads
                if attn.dim() == 4:
                    attn = attn.mean(dim=(0, 1))  # [decoder_len, encoder_len]
                elif attn.dim() == 3:
                    attn = attn.mean(dim=0)
                self.attention_list.append(attn.numpy())

    def plot_all(self):
        """Generate all interpretation plots."""
        print(f"\n[TFTInterpretation] Generating plots in {self.save_path}")

        if self.encoder_weights_list:
            self._plot_variable_importance(
                weights=np.stack(self.encoder_weights_list).mean(axis=0),
                variables=self.encoder_variables,
                title="Encoder Variable Importance",
                filename="encoder_variable_importance.png",
            )

        if self.decoder_weights_list:
            self._plot_variable_importance(
                weights=np.stack(self.decoder_weights_list).mean(axis=0),
                variables=self.decoder_variables,
                title="Decoder Variable Importance",
                filename="decoder_variable_importance.png",
            )

        if self.static_weights_list:
            self._plot_variable_importance(
                weights=np.stack(self.static_weights_list).mean(axis=0),
                variables=self.static_variables,
                title="Static Variable Importance",
                filename="static_variable_importance.png",
            )

        if self.attention_list:
            self._plot_attention_heatmap(
                attention=np.stack(self.attention_list).mean(axis=0),
                filename="attention_heatmap.png",
            )
            self._plot_attention_curve(
                attention=np.stack(self.attention_list).mean(axis=0),
                filename="attention_temporal_pattern.png",
            )

        print(f"[TFTInterpretation] ✅ All plots saved.")

    def _plot_variable_importance(
        self,
        weights: np.ndarray,
        variables: List[str],
        title: str,
        filename: str,
    ):
        """Horizontal bar chart of variable importance."""
        # Handle shape mismatch: use actual weights length
        weights = np.atleast_1d(weights.flatten())
        n_weights = len(weights)


        if n_weights == 0:
            print(f"  Skipped {filename} — no weights available.")
            return

        # If more variables than weights, trim variable names
        # If more weights than variables, label extras as "var_i"
        if n_weights <= len(variables):
            variables = variables[:n_weights]
        else:
            # n_vars = len(variables)
            variables = variables + [f"var_{i}" for i in range(len(variables), n_weights)]

        n = len(variables)

        # Normalize to percentages
        weights = weights / (weights.sum() + 1e-8) * 100

        # Sort by importance
        order = np.argsort(weights)
        weights_sorted = weights[order]
        variables_sorted = [variables[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.4)))
        bars = ax.barh(range(n), weights_sorted, color="steelblue", alpha=0.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(variables_sorted, fontsize=9)
        ax.set_xlabel("Importance (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, weights_sorted):
            if val > 1.0:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.save_path + filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")

    def _plot_attention_heatmap(self, attention: np.ndarray, filename: str):
        """
        Heatmap: rows = prediction steps, columns = attended encoder steps.
        Shows which past timesteps each future prediction attends to.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        # attention shape: [decoder_len, encoder_len]
        im = ax.imshow(attention, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_xlabel("Encoder Step (past → present)")
        ax.set_ylabel("Decoder Step (prediction horizon)")
        ax.set_title("TFT Attention — Which Past Steps Inform Each Prediction")

        # Convert steps to hours for readability
        enc_len = attention.shape[1]
        dec_len = attention.shape[0]

        # X-axis: encoder steps (hours before prediction)
        x_ticks = np.linspace(0, enc_len - 1, min(8, enc_len)).astype(int)
        x_labels = [f"-{(enc_len - t) * 0.25:.0f}h" for t in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        # Y-axis: decoder steps (hours ahead)
        y_ticks = np.linspace(0, dec_len - 1, min(6, dec_len)).astype(int)
        y_labels = [f"+{(t + 1) * 0.25:.1f}h" for t in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        plt.savefig(self.save_path + filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    def _plot_attention_curve(self, attention: np.ndarray, filename: str):
        """
        Line plot: average attention across all prediction steps.
        Shows overall temporal attention pattern — expect peak near recent past.
        """
        # Average across decoder steps → [encoder_len]
        avg_attention = attention.mean(axis=0)
        enc_len = len(avg_attention)
        hours_back = np.arange(enc_len) * 0.25  # 15-min steps → hours

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hours_back, avg_attention, color="darkorange", linewidth=1.5)
        ax.fill_between(hours_back, avg_attention, alpha=0.15, color="darkorange")
        ax.set_xlabel("Hours in Look-Back Window (0 = oldest)")
        ax.set_ylabel("Average Attention Weight")
        ax.set_title("TFT Temporal Attention Pattern (averaged over all prediction steps)")
        ax.grid(True, alpha=0.3)

        # Mark the most recent hours
        ax.axvline(x=hours_back[-1] - 6, color="gray", linestyle="--",
                   alpha=0.5, label="Last 6h")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.save_path + filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")
