from typing import Tuple

import numpy as np
import pytest


@pytest.fixture
def square_mask() -> Tuple[np.ndarray, Tuple[int, int]]:
    shape = [16, 16]
    mask = np.zeros(shape, dtype='uint8')
    mask[:shape[0] // 2, :shape[1] // 2] = 50
    mask[shape[0] // 4:shape[0] // 2, shape[1] // 4:shape[1] // 2] = 100
    return mask
