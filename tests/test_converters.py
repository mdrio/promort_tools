import cv2
import numpy as np
import pytest
from PIL import Image

from promort_tools.converters.mask_to_shapes import convert_to_shapes


@pytest.mark.parametrize('scale_factor', [1, 2, 4])
def test_mask_to_shapes_th_0(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    shapes = convert_to_shapes(square_mask, orig_res, 0)
    assert len(shapes) == 1
    coordinates = sorted(shapes[0]['coordinates'])
    print(coordinates)
    assert len(coordinates) == 5
    assert (0, 0) in coordinates
    assert (0, orig_res[1] - 1) in coordinates
    assert (orig_res[0] - 1, 0) in coordinates
    assert (orig_res[0] - 1, orig_res[1] - 1) in coordinates


@pytest.mark.parametrize('scale_factor', [1, 2, 4])
def test_mask_to_shapes_th_50(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    shapes = convert_to_shapes(square_mask, orig_res, 50)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    assert (0, 0) in coordinates
    assert (0, orig_res[1] // 2 - 1) in coordinates
    assert (orig_res[0] // 2 - 1, orig_res[1] // 2 - 1) in coordinates
    assert (orig_res[0] // 2 - 1, 0) in coordinates


@pytest.mark.parametrize('scale_factor', [1, 2, 4, 8])
def test_mask_to_shapes_th_100(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    shapes = convert_to_shapes(square_mask, orig_res, 100)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    assert (orig_res[0] // 4, orig_res[1] // 4) in coordinates
    assert (orig_res[0] // 4, orig_res[1] // 2 - 1) in coordinates
    assert (orig_res[0] // 2 - 1, orig_res[1] // 2 - 1) in coordinates
    assert (orig_res[0] // 2 - 1, orig_res[0] // 4) in coordinates
    assert (orig_res[0] // 4, orig_res[1] // 4) in coordinates


@pytest.mark.parametrize('scale_factor', [1, 2, 4, 8])
def test_rhombus_to_shapes_th_100(rhombus_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in rhombus_mask.shape]

    resized_mask = rhombus_mask.repeat(scale_factor, 0).repeat(scale_factor, 1)
    #  matprint(rhombus_mask)
    #  matprint(resized_mask)
    contours, _ = cv2.findContours(resized_mask,
                                   mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    c = contours
    shapes = convert_to_shapes(rhombus_mask, orig_res, 100)
    contours = contours[0].reshape(contours[0].shape[0], 2)
    contours = list([(float(_[0]), float(_[1])) for _ in contours])
    shape = shapes[0]['coordinates']
    #  print('contours', sorted(contours))
    #  print('shape', sorted(shape))
    #  print(c[0].shape)
    #  s = np.array(shape, dtype='int32')
    #  s = np.expand_dims(s, 1)
    #
    #  print(s.shape)
    #
    #  image = np.zeros((200, 200, 3), dtype='uint8')
    #  cv2.fillPoly(image, pts=[s], color=(255, 255, 255))
    #  cv2.fillPoly(image, pts=[c[0]], color=(255, 0, 0))
    #  Image.fromarray(image).show()
    assert set(shape) - set(contours) == set()


def matprint(mat, fmt="g"):
    col_maxes = [
        max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T
    ]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")
