''' Different callbacks that can be plugged into the callback decorators.'''

from typing import Dict, Any
import pytorch_lightning as pl

def cash_task_info(module: pl.LightningModule, results: Dict, cache: Dict[str, Any]) -> Dict[str, Any]:
    adaptation_gain = results["train_losses"][0] - results["train_errors"][0]
    cache['adaptation_gain'] = adaptation_gain
    return cache
