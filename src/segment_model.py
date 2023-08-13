import typing as tp

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.config import Config
from src.losses import get_losses
from src.metrics import get_classification_metrics, get_segmentation_metrics
from src.io import load_object


class BarcodeModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self._cfg = cfg
        self.model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            classes=cfg.data_config.n_classes,
            aux_params={
                'pooling': 'avg', 
                'dropout': 0.2, 
                'classes': cfg.data_config.n_classes
            },
        )
        self._cls_losses = get_losses(self._cfg.cls_losses)
        self._seg_losses = get_losses(self._cfg.seg_losses)
        cls_metrics = get_classification_metrics(
            num_classes=cfg.data_config.n_classes,
            num_labels=cfg.data_config.n_classes,
            task='binary',
            average='macro',
            threshold=0.7,
        )
        seg_metrics = get_segmentation_metrics(labels=['', '1', '2', '3', '4'])  # названий нет, поэтому обозначим цифрами
        self._val_cls_metrics = cls_metrics.clone(prefix='val_')
        self._test_cls_metrics = cls_metrics.clone(prefix='test_')
        self._val_seg_metrics = seg_metrics.clone(prefix='val_')
        self._test_seg_metrics = seg_metrics.clone(prefix='test_')
        self.save_hyperparameters(self._cfg.dict())

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._cfg.optimizer)(
            self.model.parameters(), lr=self._cfg.lr, **self._cfg.optimizer_kwargs,
        )
        scheduler = load_object(self._cfg.scheduler)(optimizer, **self._cfg.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._cfg.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, gt_masks = batch
        pred_masks_logits, pred_labels_logits = self(images)
        return self._calculate_loss(pred_masks_logits, pred_labels_logits, gt_masks, gt_targets, 'train_')

    def validation_step(self, batch, batch_idx):
        images, gt_masks, gt_targets = batch
        pred_masks_logits, pred_labels_logits = self(images)
        self._calculate_loss(
            pred_masks_logits, 
            pred_labels_logits, 
            gt_masks, 
            # gt_targets, 
            'val_'
        )
        pred_masks = torch.sigmoid(pred_masks_logits)
        pred_labels = torch.sigmoid(pred_labels_logits)
        # self._val_cls_metrics(pred_labels, gt_targets)
        self._val_seg_metrics(pred_masks, gt_masks)

    def test_step(self, batch, batch_idx):
        images, gt_masks, gt_targets = batch
        pred_masks_logits, pred_labels_logits = self(images)
        pred_masks = torch.sigmoid(pred_masks_logits)
        pred_labels = torch.sigmoid(pred_labels_logits)
        # self._test_cls_metrics(pred_labels, gt_targets)
        self._test_seg_metrics(pred_masks, gt_masks)

    def on_validation_epoch_start(self) -> None:
        # self._val_cls_metrics.reset()
        self._val_seg_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        # self.log_dict(self._val_cls_metrics.compute(), on_epoch=True, on_step=False)
        self.log_dict(self._val_seg_metrics.compute(), on_epoch=True, on_step=False)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_cls_metrics.compute(), on_epoch=True, on_step=False)
        self.log_dict(self._test_seg_metrics.compute(), on_epoch=True, on_step=False)

    def _calculate_loss(
        self,
        pred_masks_logits: torch.Tensor,
        pred_labels_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        # gt_targets: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        # total_loss = 0
        # for seg_loss in self._seg_losses:
        #     loss = seg_loss.loss(pred_masks_logits, gt_masks)
        #     total_loss += seg_loss.weight * loss
        #     self.log(f'{prefix}{seg_loss.name}_loss', loss.item())
        # for cls_loss in self._cls_losses:
        #     loss = cls_loss.loss(pred_labels_logits, gt_targets)
        #     total_loss += cls_loss.weight * loss
        #     self.log(f'{prefix}{cls_loss.name}_loss', loss.item())
        total_loss = loss
        self.log(f'{prefix}total_loss', total_loss.item())
        # return total_loss
