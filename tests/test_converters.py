import pytest

from promort_tools.converters.mask_to_shapes import convert_to_shapes


@pytest.mark.parametrize('scale_factor', [2, 4])
def test_mask_to_shapes_th_0(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    shapes = convert_to_shapes(square_mask, orig_res, 0)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    assert coordinates[0] == (0, 0)
    assert coordinates[1] == (0, orig_res[1] - 1)
    assert coordinates[2] == (orig_res[0] - 1, orig_res[1] - 1)
    assert coordinates[3] == (orig_res[0] - 1, 0)
    assert coordinates[4] == (0, 0)


@pytest.mark.parametrize('scale_factor', [2, 4])
def test_mask_to_shapes_th_50(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    shapes = convert_to_shapes(square_mask, orig_res, 50)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    assert coordinates[0] == (0, 0)
    assert coordinates[1] == (0, orig_res[1] // 2 - 1)
    assert coordinates[2] == (orig_res[0] // 2 - 1, orig_res[1] // 2 - 1)
    assert coordinates[3] == (orig_res[0] // 2 - 1, 0)
    assert coordinates[4] == (0, 0)


@pytest.mark.parametrize('scale_factor', [2, 4])
def test_mask_to_shapes_th_100(square_mask, scale_factor):
    orig_res = [_ * scale_factor for _ in square_mask.shape]
    shapes = convert_to_shapes(square_mask, orig_res, 100)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    assert coordinates[0] == (orig_res[0] // 4, orig_res[1] // 4)
    assert coordinates[1] == (orig_res[0] // 4, orig_res[1] // 2 - 1)
    assert coordinates[2] == (orig_res[0] // 2 - 1, orig_res[1] // 2 - 1)
    assert coordinates[3] == (orig_res[0] // 2 - 1, orig_res[0] // 4)
    assert coordinates[4] == (orig_res[0] // 4, orig_res[1] // 4)
