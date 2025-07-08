from dataclasses import dataclass
from typing import Tuple

@dataclass
class TransformConfig:
    resize_height: int
    resize_width: int
    hflip_prob: float
    vflip_prob: float
    color_jitter: Tuple[float]


@dataclass
class DataConfig:
    coco_dir: str # MSCOCO dataset root directory
    city_dir: str # CityScape dataset root directory
    ade20k_dir: str
    pascal_dir: str
    transform: TransformConfig
    batch_size: int