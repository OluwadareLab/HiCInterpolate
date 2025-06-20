from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PathsConfig:
    root_dir: str
    data_dir: str
    record_file: str
    image_dir: str
    vgg_model_file: str

    model_file: str
    model_state_dir: str

    output_dir: str
    img_val_plot_path: str
    train_val_plot_file: str
    test_plot_file: str
    eval_metrics_file: str
    psnr_eval_plot_file: str
    ssim_eval_plot_file: str
    lr_file: str


@dataclass
class DataConfig:
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
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    eval: EvalConfig
