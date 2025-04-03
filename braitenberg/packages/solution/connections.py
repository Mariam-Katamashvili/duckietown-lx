from typing import Tuple

import numpy as np


import math

import numpy as np
from typing import Tuple

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mid = w / 2.0

    rows = np.arange(h).reshape(h, 1)   
    cols = np.arange(w).reshape(1, w)    

    vertical_weight = rows / float(h)
    
    directional = (cols - mid) / mid

    sigma = w * 0.15  
    center_mask = np.exp(-((cols - mid) ** 2) / (2 * sigma ** 2))
    
    left_matrix = vertical_weight * directional * center_mask
    return left_matrix.astype(np.float32)

def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    return np.fliplr(get_motor_left_matrix(shape)).astype(np.float32)