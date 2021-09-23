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


@pytest.fixture
def rhombus_mask() -> Tuple[np.ndarray, Tuple[int, int]]:
    shape = [16, 16]
    #  mask = np.zeros(shape, dtype='uint8')
    #  for i in range(9):
    #      j = i if i < 4 else 8 - i
    #
    #      mask[8 - j:8 + j + 1, 3 + i + 1] = 100

    mask = np.zeros(shape, dtype='uint8')
    diag = 4
    for i in range(diag):
        j = i if i < diag // 2 else diag - 1 - i

        mask[diag - 1 - j:diag + j, i] = 100
    return mask
