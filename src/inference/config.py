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
    inference: str
    inference_raw: str
    vgg_model: str
    model: str
    hic_map: str
    metrics: str
    log: str


@dataclass
class DataConfig:
    patch: int
    interpolator_images_map: Dict[str, str]
    batch_size: int


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
    model: ModelConfig
    loss: LossConfig
