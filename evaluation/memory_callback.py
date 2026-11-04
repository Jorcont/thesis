import torch
import psutil
import os
import pytorch_lightning as pl

class MemoryTrackerCallback(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        # CPU RAM
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9

        # GPU VRAM
        if torch.cuda.is_available():
            gpu_alloc  = torch.cuda.memory_allocated()  / 1e9
            gpu_reserv = torch.cuda.memory_reserved()   / 1e9
            gpu_peak   = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()  # reset peak after each epoch
        else:
            gpu_alloc = gpu_reserv = gpu_peak = 0.0

        print(
            f"\n[MEM] Epoch {trainer.current_epoch} end:\n"
            f"  CPU RAM   : {rss_gb:.2f} GB use this for SLURM --mem=\n"
            f"  GPU alloc : {gpu_alloc:.2f} GB\n"
            f"  GPU reserv: {gpu_reserv:.2f} GB\n"
            f"  GPU peak  : {gpu_peak:.2f} GB  ← use this for SLURM --mem-per-gpu"
        )