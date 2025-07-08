from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DirConfig:
    root: str
    data: str
    image: str
    model_state: str
    output: str


@dataclass
class FileConfig:
    train: str
    test: str
    vgg_model: str
    snapshot: str
    model: str
    train_val_plot: str
    test_hic_map: str
    eval_metrics: str
    psnr_eval_plot: str
    ssim_eval_plot: str
    lr_plot: str
    log: str


@dataclass
class DataConfig:
    patch: int
    interpolator_images_map: Dict[str, str]
    train_val_ratio: List[float]
    batch_size: int


@dataclass
class TrainingConfig:
    epochs: int
    save_every: int
    learning_rate: float
    decay_steps: int
    decay_rate: float
    lr_staircase: bool


@dataclass
class FlowConfig:
    num_of_convs: List[int]
    out_channels: List[int]


@dataclass
class ModelConfig:
    name: str
    init_in_channels: int
    init_out_channels: int
    pyramid_level: int
    ext_feature_level: int
    unique_levels: int
    flow: FlowConfig
    fusion_pyramid_level: int


@dataclass
class LossWeightConfig:
    name: str
    boundaries: List[int]
    values: List[float]


@dataclass
class LossConfig:
    weight_parameters: List[LossWeightConfig]


@dataclass
class EvalConfig:
    num_visualization_samples: int


@dataclass
class Config:
    device: str
    dir: DirConfig
    file: FileConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    eval: EvalConfig
