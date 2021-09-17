from promort_tools.converters.mask_to_shapes import convert_to_shapes


def test_mask_to_shapes_th_0(mask_data):
    mask, orig_res = mask_data
    shapes = convert_to_shapes(mask, orig_res, -1)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    print(coordinates)
    assert coordinates[0] == (0, 0)
    assert coordinates[1] == (0, orig_res[1] - 2)
    assert coordinates[2] == (orig_res[0] - 2, orig_res[1] - 2)
    assert coordinates[3] == (orig_res[0] - 2, 0)
    assert coordinates[4] == (0, 0)


def test_mask_to_shapes_th_100(mask_data):
    mask, orig_res = mask_data
    shapes = convert_to_shapes(mask, orig_res, 100)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert len(coordinates) == 5
    print(coordinates)
    assert coordinates[0] == (0, 0)
    assert coordinates[1] == (0, orig_res[1] // 2 - 2)
    assert coordinates[2] == (orig_res[0] // 2 - 2, orig_res[1] // 2 - 2)
    assert coordinates[3] == (orig_res[0] // 2 - 2, 0)
    assert coordinates[4] == (0, 0)
