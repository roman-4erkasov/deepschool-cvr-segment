import typing as tp

import torch
from torchmetrics import Metric, MetricCollection, F1Score, Precision, Recall
from segmentation_models_pytorch.metrics import get_stats


def get_classification_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'f1': F1Score(**kwargs),
        'precision': Precision(**kwargs),
        'recall': Recall(**kwargs),
    })


def get_segmentation_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'iou': IoUMultiLabel(**kwargs),
    })


class IoUMultiLabel(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, labels: tp.List[str], threshold: float = 0.5):
        super().__init__()
        self._labels = labels
        self._threshold = threshold
        self._init_logical_values()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        target = target.long()
        for idx, label in enumerate(self._labels):
            tp, fp, fn, _ = get_stats(
                preds.narrow(dim=1, start=idx, length=1),
                target.narrow(dim=1, start=idx, length=1),
                mode='binary',
                threshold=self._threshold,
            )
            self.tp += tp.sum()
            self.fp += fp.sum()
            self.fn += fn.sum()
            setattr(self, f'tp_{label}', getattr(self, f'tp_{label}') + tp.sum())
            setattr(self, f'fp_{label}', getattr(self, f'fp_{label}') + fp.sum())
            setattr(self, f'fn_{label}', getattr(self, f'fn_{label}') + fn.sum())

    def compute(self) -> tp.Dict[str, torch.Tensor]:
        result = {'iou': self.tp / (self.tp + self.fp + self.fn)}
        for label in self._labels:
            tp = getattr(self, f'tp_{label}')
            fp = getattr(self, f'fp_{label}')
            fn = getattr(self, f'fn_{label}')
            result[f'iou_{label}'] = tp / (tp + fp + fn)
        return result

    def _init_logical_values(self):
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')
        for idx, label in enumerate(self._labels):
            self.add_state(f'tp_{label}', default=torch.tensor(0), dist_reduce_fx='sum')
            self.add_state(f'fp_{label}', default=torch.tensor(0), dist_reduce_fx='sum')
            self.add_state(f'fn_{label}', default=torch.tensor(0), dist_reduce_fx='sum')
