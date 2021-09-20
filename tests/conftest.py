from typing import Tuple

import numpy as np
import pytest


@pytest.fixture
def square_mask() -> Tuple[np.ndarray, Tuple[int, int]]:
    orig_res = np.array([32, 32])
    level_downsample = 2
    shape = orig_res // level_downsample
    mask = np.zeros(shape, dtype='uint8')
    mask[:shape[0] // 2, :shape[1] // 2] = 50
    mask[shape[0] // 4:shape[0] // 2, shape[1] // 4:shape[1] // 2] = 100
    return mask
