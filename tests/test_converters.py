from promort_tools.converters.mask_to_shapes import convert_to_shapes


def test_mask_to_roi_th_0(mask_data):
    mask, orig_res = mask_data
    shapes = convert_to_shapes(mask, orig_res, -1)
    print(shapes)
    assert len(shapes) == 1
    coordinates = shapes[0]['coordinates']
    assert coordinates[0] == (0, 0)
    assert coordinates[0] == (0, orig_res[1] - 1)
    assert coordinates[0] == (orig_res[0] - 1, orig_res[1] - 1)
    assert coordinates[0] == (orig_res[0] - 1, 0)
