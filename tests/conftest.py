from typing import Tuple

import numpy as np
import pytest


@pytest.fixture
def mask_data() -> Tuple[np.ndarray, Tuple[int, int]]:
    orig_res = np.array([32, 32])
    level_downsample = 2
    shape = orig_res // level_downsample
    mask = np.zeros(shape, dtype='uint8')
    mask[:shape[0] // 2, :shape[1] // 2] = 100
    return mask, orig_res
