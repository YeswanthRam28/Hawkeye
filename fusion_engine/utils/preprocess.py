# fusion_engine/utils/preprocess.py

import numpy as np


class Preprocessor:
    """Basic preprocessing for sensor fusion pipeline."""

    def __init__(self, normalize=True, clip=True):
        self.normalize = normalize
        self.clip = clip

    def normalize_value(self, x, min_val=0, max_val=100):
        """Scale value into 0–1 range."""
        if not self.normalize:
            return x
        return (x - min_val) / (max_val - min_val)

    def clip_value(self, x, min_val=0, max_val=1):
        """Remove extreme outliers."""
        if not self.clip:
            return x
        return np.clip(x, min_val, max_val)

    def moving_average(self, values, window=3):
        """Smooth using simple moving average."""
        if len(values) < window:
            return np.mean(values)
        return np.mean(values[-window:])

    def process(self, value):
        """Full pipeline: normalize → clip."""
        value = self.normalize_value(value)
        value = self.clip_value(value)
        return value
