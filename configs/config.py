from dataclasses import dataclass, field
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
    snapshot: str
    model: str
    test_hic_map: str
    val_metrics: str
    num_visualization_samples: int
    psnr_val_plot: str
    ssim_val_plot: str
    scc_val_plot: str
    hicrep_val_plot: str
    genome_disco_val_plot: str
    lpips_val_plot: str
    train_val_loss_plot: str
    lr_plot: str
    log: str
    val_metrics_plot: str


@dataclass
class DataConfig:
    patch: int
    interpolator_images_map: Dict[str, str]
    train_val_test_ratio: List[float]
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
class DiffusionConfig:
    enabled: bool = True
    timesteps: int = 64
    beta_start: float = 0.0001
    beta_end: float = 0.02
    hidden_channels: int = 32
    inference_timestep: int = 4
    preserve_input_support: bool = True


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
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


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
    scc: bool = True
    hicrep: bool = True
    genome_disco: bool = True
    lpips: bool = True


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
