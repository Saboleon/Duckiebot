from typing import Tuple
import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Left motor weight matrix: highest at bottom-left, decreasing toward top-right."""
    height, width = shape
    x_gradient = np.linspace(1.0, 0.1, width)
    y_gradient = np.linspace(0.1, 1.0, height)
    xv, yv = np.meshgrid(x_gradient, y_gradient)
    return xv * yv

def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Right motor weight matrix: highest at bottom-right, decreasing toward top-left."""
    left_matrix = get_motor_left_matrix(shape)
    return np.fliplr(left_matrix)
