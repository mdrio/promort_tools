import cv2
import pytest

from promort_tools.converters.mask_to_shapes import convert_to_shapes, BasicScaler


@pytest.mark.parametrize("scale_factor", [1, 2, 4, 8])
def test_mask_to_shapes_th_0(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    scaler = BasicScaler(square_mask.shape, orig_res)
    shapes = convert_to_shapes(square_mask, 0, scaler)["shapes"]
    assert len(shapes) == 1
    coordinates = sorted(shapes[0]["coordinates"])
    print(coordinates, orig_res)
    assert len(coordinates) == 5
    assert (scale_factor / 2,) * 2 in coordinates
    assert (scale_factor / 2, orig_res[1] - scale_factor / 2) in coordinates
    assert (orig_res[0] - scale_factor / 2, scale_factor / 2) in coordinates
    assert (
        orig_res[0] - scale_factor / 2,
        orig_res[1] - scale_factor / 2,
    ) in coordinates


@pytest.mark.parametrize("scale_factor", [1, 2, 4, 8])
def test_mask_to_shapes_th_50(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    scaler = BasicScaler(square_mask.shape, orig_res)
    shapes = convert_to_shapes(square_mask, 50, scaler)["shapes"]
    assert len(shapes) == 1
    coordinates = shapes[0]["coordinates"]
    assert len(coordinates) == 5
    assert (scale_factor / 2,) * 2 in coordinates
    assert (scale_factor / 2, orig_res[1] / 2 - scale_factor / 2) in coordinates
    assert (
        orig_res[0] / 2 - scale_factor / 2,
        orig_res[1] / 2 - scale_factor / 2,
    ) in coordinates
    assert (orig_res[0] / 2 - scale_factor / 2, scale_factor / 2) in coordinates


@pytest.mark.parametrize("scale_factor", [1, 2, 4, 8])
def test_mask_to_shapes_th_100(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    scaler = BasicScaler(square_mask.shape, orig_res)
    shapes = convert_to_shapes(square_mask, 100, scaler)["shapes"]
    assert len(shapes) == 1
    coordinates = shapes[0]["coordinates"]
    assert len(coordinates) == 5

    assert (
        orig_res[0] / 4 + scale_factor / 2,
        orig_res[1] / 4 + scale_factor / 2,
    ) in coordinates

    assert (
        orig_res[0] / 4 + scale_factor / 2,
        orig_res[1] / 2 - scale_factor / 2,
    ) in coordinates

    assert (
        orig_res[0] / 2 - scale_factor / 2,
        orig_res[1] / 2 - scale_factor / 2,
    ) in coordinates

    assert (
        orig_res[0] / 2 - scale_factor / 2,
        orig_res[0] / 4 + scale_factor / 2,
    ) in coordinates
