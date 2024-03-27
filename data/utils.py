from pathlib import Path
import numpy as np
from data.transforms import transforms as tf
from typing import List

def pair_data(path: List[str]) -> np.ndarray:
    return np.array([[file for file in Path(p).iterdir()] for p in path]).T

def unpaired_data(path: str) -> np.ndarray:
    return np.array([file for file in Path(path).iterdir()])

def vision_2_vision(data: np.ndarray | Path) -> list:
    if isinstance(data, np.ndarray):
        return [tf(np.load(point)) for point in data]
    if isinstance(data, Path):
        return tf(np.load(data))