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
    dataset_dict: str
    vgg_model: str
    snapshot: str
    model: str
    test_hic_map: str
    val_metrics: str
    num_visualization_samples: int
    psnr_val_plot: str
    ssim_val_plot: str
    scc_val_plot: str
    pcc_val_plot: str
    genome_disco_val_plot: str
    ncc_val_plot: str
    lpips_val_plot: str
    train_val_loss_plot: str
    grad_norm_plot: str
    lr_plot: str
    log: str


@dataclass
class DataConfig:
    patch: int
    interpolator_images_map: Dict[str, str]
    train_val_test_ratio: List[float]
    batch_size: int


@dataclass
class TrainingConfig:
    epochs: int
    restart_every: int
    save_every: int
    init_lr: float
    min_lr: float
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
class Config:
    device: str
    dir: DirConfig
    file: FileConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
