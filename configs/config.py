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
    hicrep_val_plot: str
    ent3c_val_plot: str
    lr_plot: str
    log: str


@dataclass
class DataConfig:
    patch: int
    interpolator_images_map: Dict[str, str]
    train_val_test_ratio: List[float]
    batch_size: int
    apply_log1p: bool = True
    signed_log1p: bool = False
    apply_normalization: bool = True
    normalization_mode: str = "train_percentile"
    normalization_lower_percentile: float = 0.1
    normalization_upper_percentile: float = 99.9
    normalization_max_files: int = 0
    normalization_sample_values_per_file: int = 4096
    normalization_fixed_low: float = 0.0
    normalization_fixed_high: float = 1.0
    norm_eps: float = 1e-8
    sparse_sampling_enabled: bool = False
    sparse_sampling_nonzero_threshold: float = 1e-6
    sparse_sampling_informative_ratio: float = 0.02
    sparse_sampling_informative_boost: float = 3.0


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
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    scheduler_name: str = "reduce_on_plateau"
    warmup_epochs: int = 5
    plateau_factor: float = 0.5
    plateau_patience: int = 8
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 2
    lr_metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "genome_disco": 0.5,
        "hicrep": 0.5,
        "ssim": 0.0,
    })
    best_model_metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "genome_disco": 0.45,
        "hicrep": 0.45,
        "ssim": 0.10,
    })
    best_model_metric_norm_eps: float = 1e-8


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
    warp_mode: str = "nearest"
    flow_upsample_mode: str = "nearest"


@dataclass
class LossWeightConfig:
    name: str
    boundaries: List[int]
    values: List[float]


@dataclass
class LossConfig:
    weight_parameters: List[LossWeightConfig]
    sparse_weight_enabled: bool = False
    sparse_weight_nonzero_threshold: float = 1e-6
    sparse_weight_nonzero_boost: float = 4.0


@dataclass
class Config:
    device: str
    dir: DirConfig
    file: FileConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
