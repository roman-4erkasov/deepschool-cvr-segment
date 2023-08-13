import typing as tp
from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    batch_size: int
    n_workers: int
    train_size: float
    val_size: float
    img_width: int
    img_height: int
    # ann_train: str
    # ann_val: str
    # ann_test: str
    images_path: str
    scopes_path: str
    data_path: str
    n_classes: int


class Config(BaseModel):
    project_name: str
    task_name: str
    data_config: DataConfig
    n_epochs: int
    lr: float
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    cls_losses: tp.Optional[tp.List[LossConfig]] = None
    seg_losses: tp.Optional[tp.List[LossConfig]] = None
    task: str  # "detection" or "segmentation"


    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
