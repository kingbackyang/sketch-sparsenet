from typing import Any, Dict

import numpy as np
import torch
from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback

__all__ = ['MeanIoU']


class MeanIoU(Callback):

    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor

    def _before_epoch(self) -> None:
        self.total_seen = 0.0
        self.total_correct = [0, 0, 0]
        self.total_positive = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        # outputs = outputs[targets != self.ignore_label]
        # targets = targets[targets != self.ignore_label]
        topk = (1, 5, 10)
        self.total_seen += targets.size(0)

        for i, k in enumerate(topk):
            self.total_correct[i] += outputs[:, :k].eq(targets.view(-1, 1)).sum().item()

    def _after_epoch(self) -> None:
        res = [correct / self.total_seen for correct in self.total_correct]
        print("total samples", self.total_seen, "top1 acc:", res[0], "top5 acc: ", res[1], "top10 acc:", res[2])

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar("top1 acc", res[0] * 100)
            self.trainer.summary.add_scalar("top5 acc", res[1] * 100)
            self.trainer.summary.add_scalar("top10 acc", res[2] * 100)
        else:
            print(res[0])
