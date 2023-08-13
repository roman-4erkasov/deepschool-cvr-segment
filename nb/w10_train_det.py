import sys
sys.path.append("..")
import math
import torch
import torch as th
from PIL import Image
import numpy as np

import torchvision as thv
# import torchmetrics as thm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import pytorch_lightning as pl

from src.datamodule import BarcodeDM
from src.config import Config
import albumentations as albu



cfg = Config.from_yaml("../config/baseline_detect.yml")
data = BarcodeDM(cfg.data_config, task=cfg.task, dry_run=True)
data.prepare_data()
data.setup()



class DetectModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = thv.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        self.val_map = MeanAveragePrecision()
        self.test_map = MeanAveragePrecision()
    
    def forward(self, x: torch.Tensor):
        print(f"forward:")
        return self.model(x)
    
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.005, momentum=0.9, weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
            
    def training_step(self, batch, batch_idx):
        """
        """
        print(f"training_step:begin:{len(batch)=} {batch_idx=}")
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # return loss
        print(f"training_step:end:{loss=}")
        return {
            'loss': loss, 
            'log': loss_dict, 
            'progress_bar': loss_dict
        }

    def validation_step(self, batch, batch_idx):
        print(f"validation_step:begin:{len(batch)=} {batch_idx=}")
        images, targets = batch
        with th.no_grad():
            pred = self.model(images)
        print(f"validation_step:{pred[0]['boxes'][:10]=}")
        print(f"validation_step:{targets[0]['boxes']=}")
        self.val_map.update(
            preds=pred,target=targets
        )
        self.log_dict(self.val_map.compute(), on_step=False, on_epoch=True,prog_bar=False)
        print(f"validation_step:end")

    def test_step(self, batch, batch_idx):
        print(f"test_step:begin:{len(batch)=} {batch_idx=}")
        images, targets = batch
        with th.no_grad():
            pred = self.model(images)
        # self.log(
        #     "test_iou", 
        #     thv.ops.box_iou(
        #         th.stack([t["boxes"] for t in pred ]).squeeze(), 
        #         th.stack([t["boxes"] for t in targets ]).squeeze()
        #     )
        # )
        print(f"test_step:{pred[0]['boxes'][:10]=}")
        print(f"test_step:{targets[0]['boxes']=}")
        self.test_map.update(preds=pred, target=targets)
        self.log_dict(self.test_map.compute(), on_step=True, on_epoch=True,prog_bar=False)
        print(f"test_step:end")

    # def on_validation_epoch_start(self) -> None:
    #     pass

    def on_validation_epoch_end(self) -> None:
        print(f"on_validation_epoch_end:begin")
        # self.log_dict(self._val_cls_metrics.compute(), on_epoch=True, on_step=False)
        # self.log_dict(self._val_seg_metrics.compute(), on_epoch=True, on_step=False)
        print(self.val_map.compute())
        print(f"on_validation_epoch_end:end")

    def on_test_epoch_end(self) -> None:
        print(f"on_test_epoch_end:begin")
        # self.log_dict(self._test_cls_metrics.compute(), on_epoch=True, on_step=False)
        # self.log_dict(self._test_seg_metrics.compute(), on_epoch=True, on_step=False)
        print(self.test_map.compute())
        print(f"on_test_epoch_end:begin")

    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     optimizer.step()
    #     # self.lr_scheduler.step()  # Step per iteration
    # def configure_optimizers(self):
    #     print(f"configure_optimizers:")
    #     return th.optim.Adam(self.parameters(), lr=0.0001)


model = DetectModel(cfg)
trainer = pl.Trainer(
    max_epochs=20,
    # accelerator=config.accelerator,
    # devices=[config.device],
    # callbacks=[
    #     checkpoint_callback,
    #     EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
    #     LearningRateMonitor(logging_interval='epoch'),
    # ],
)
trainer.fit(model=model, datamodule=data)

