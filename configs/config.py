from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DirConfig:
    data: str
    image: str
    model_state: str
    output: str


@dataclass
class FileConfig:
    dataset_dict: str
    snapshot: str
    model: str
    test_hic_map: str
    val_metrics: str
    num_visualization_samples: int
    psnr_val_plot: str
    ssim_val_plot: str
    ms_ssim_val_plot: str
    hicrep_val_plot: str
    train_val_loss_plot: str
    lr_plot: str
    log: str


@dataclass
class DataConfig:
    patch: int
    resolution: int
    interpolator_images_map: Dict[str, str]
    batch_size: int


@dataclass
class TrainingConfig:
    epochs: int
    save_every: int
    lr: float
    min_lr: float
    weight_decay: float = 5e-5
    warmup_epochs: int = 3
    grad_clip: float = 1.0


@dataclass
class FlowConfig:
    num_of_convs: List[int]
    out_channels: List[int]


@dataclass
class ModelConfig:
    name: str


@dataclass
class LossWeightConfig:
    name: str
    boundaries: List[int]
    values: List[float]


@dataclass
class LossConfig:
    weight_parameters: List[LossWeightConfig]


@dataclass
class EvalMetricConfig:
    psnr: bool = True
    ssim: bool = True
    hicrep: bool = True



@dataclass
class EvaluationConfig:
    monitor: str = "ssim"
    metrics: EvalMetricConfig = field(default_factory=EvalMetricConfig)


@dataclass
class Config:
    device: str
    dir: DirConfig
    file: FileConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
