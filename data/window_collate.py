''' Window-level collate for the supervised baseline.'''
import torch


def make_window_collate(windows_per_step: int, use_query: bool = True):
    """Pools all windows of the sampled tasks and draws exactly
    `windows_per_step` windows. Yields (X, y) instead of a task tuple,
    so the baseline training step takes its supervised branch."""
    def collate(task_samples):
        xs, ys = [], []
        for sample in task_samples:
            X_s, y_s, X_q, y_q = sample
            parts = [(X_s, y_s)] + ([(X_q, y_q)] if use_query else [])
            for X, y in parts:
                xs.append(X if isinstance(X, (tuple, list)) else (X,))
                ys.append(y)

        n_feat = len(xs[0])
        X_pool = tuple(torch.cat([x[i] for x in xs], dim=0) for i in range(n_feat))
        y_pool = torch.cat(ys, dim=0)

        n = y_pool.shape[0]
        k = min(windows_per_step, n)
        idx = torch.randperm(n)[:k]
        X_batch = tuple(x[idx] for x in X_pool)
        return (X_batch if n_feat > 1 else X_batch[0]), y_pool[idx]
    return collate