from typing import Tuple
import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    alpha = 2 * np.pi / resolution
    delta_ticks = ticks - prev_ticks
    return delta_ticks * alpha, ticks

def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:
    d_left  = R * delta_phi_left
    d_right = R * delta_phi_right
 
    d = (d_right + d_left) / 2.0
    delta_theta = (d_right - d_left) / baseline
 
    theta_mid = theta_prev + delta_theta / 2.0
    x_curr = x_prev + d * np.cos(theta_mid)
    y_curr = y_prev + d * np.sin(theta_mid)
    theta_curr = np.arctan2(np.sin(theta_prev + delta_theta), np.cos(theta_prev + delta_theta))
 
    return x_curr, y_curr, theta_curr
    