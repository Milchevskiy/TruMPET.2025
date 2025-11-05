import math
import numpy as np


def gaussian_weights(point_number: int, window: np.ndarray) -> np.ndarray:
    """Векторная версия exponent_factor для массива сдвигов"""
    argument = 6 * np.abs(window.astype(np.float32)) / point_number
    return np.exp(-0.5 * argument**2)
