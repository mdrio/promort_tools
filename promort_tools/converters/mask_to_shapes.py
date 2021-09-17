#  Copyright (c) 2021, CRS4
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the "Software"), to deal in
#  the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#  the Software, and to permit persons to whom the Software is furnished to do so,
#  subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import logging
import sys
from math import log, sqrt
from typing import Dict, Tuple

import cv2
import numpy as np
import zarr
from shapely.affinity import scale
from shapely.geometry import Polygon

from promort_tools.libs.utils.logger import LOG_LEVELS, get_logger

LOGGER = logging.getLogger()


def convert_to_shapes(mask: np.ndarray, original_resolution: Tuple[int, int],
                      threshold: int):
    def _apply_threshold(mask: np.ndarray, threshold: int) -> np.ndarray:
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1

    def _get_cores(mask):
        contours, _ = cv2.findContours(mask,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(None, [_contour_to_shape(c) for c in contours])
        return contours

    def _contour_to_shape(contour):
        normalize_contour = []
        for x in contour:
            normalize_contour.append(tuple(x[0]))
        try:
            return Shape(normalize_contour)
        except ValueError:
            return None

    def _filter_cores(cores, slide_area, core_min_area=0.02):
        accepted_cores = []
        for core in cores:
            if (core.get_area() * 100 / slide_area) >= core_min_area:
                accepted_cores.append(core)
        return accepted_cores

    def _get_scale_factor(slide_resolution, mask_resolution):
        scale_factor = sqrt((slide_resolution[0] * slide_resolution[1]) /
                            (mask_resolution[0] * mask_resolution[1]))
        LOGGER.info('Scale factor is %r', scale_factor)
        return scale_factor

    def _build_slide_json(cores, scale_factor):
        scale_factor = log(scale_factor, 2)
        slide_shapes = [{
            'coordinates': c.get_coordinates(scale_factor),
            'length': c.get_length(scale_factor),
            'area': c.get_area(scale_factor)
        } for c in cores]
        return slide_shapes

    _apply_threshold(mask, threshold)

    cores = _filter_cores(_get_cores(mask), mask.size)
    #  grouped_cores = self._group_nearest_cores(cores, mask.shape[0])
    scale_factor = _get_scale_factor(original_resolution, mask.shape)
    return _build_slide_json(cores, scale_factor)


class Shape:
    def __init__(self, segments):
        self.polygon = Polygon(segments)

    def get_bounds(self):
        bounds = self.polygon.bounds
        try:
            return {
                'x_min': bounds[0],
                'y_min': bounds[1],
                'x_max': bounds[2],
                'y_max': bounds[3]
            }
        except IndexError:
            raise InvalidPolygonError()

    def get_coordinates(self, scale_level=0):
        if scale_level != 0:
            polygon = self._rescale_polygon(scale_level)
        else:
            polygon = self.polygon
        return list(polygon.exterior.coords)

    def get_area(self, scale_level=0):
        if scale_level != 0:
            polygon = self._rescale_polygon(scale_level)
        else:
            polygon = self.polygon
        return polygon.area

    def get_length(self, scale_level=0):
        if scale_level != 0:
            polygon = self._rescale_polygon(scale_level)
        else:
            polygon = self.polygon
        polygon_path = np.array(polygon.exterior.coords[:])
        _, radius = cv2.minEnclosingCircle(polygon_path.astype(int))
        return radius * 2

    def _rescale_polygon(self, scale_level):
        scaling = pow(2, scale_level)
        return scale(self.polygon, xfact=scaling, yfact=scaling, origin=(0, 0))

    def get_full_mask(self, scale_level=0, tolerance=0):
        if scale_level != 0:
            polygon = self._rescale_polygon(scale_level)
            scale_factor = pow(2, scale_level)
        else:
            polygon = self.polygon
            scale_factor = 1
        if tolerance > 0:
            polygon = polygon.simplify(tolerance, preserve_topology=False)
        bounds = self.get_bounds()
        box_height = int((bounds['y_max'] - bounds['y_min']) * scale_factor)
        box_width = int((bounds['x_max'] - bounds['x_min']) * scale_factor)
        mask = np.zeros((box_height, box_width), dtype=np.uint8)
        polygon_path = polygon.exterior.coords[:]
        polygon_path = [(int(x - bounds['x_min'] * scale_factor),
                         int(y - bounds['y_min'] * scale_factor))
                        for x, y in polygon_path]
        cv2.fillPoly(mask, np.array([
            polygon_path,
        ]), 1)
        return mask


class InvalidPolygonError(Exception):
    ...


def main(argv):
    parser = _make_parser()
    args = parser.parse_args(argv)

    global LOGGER
    LOGGER = get_logger(args.log_level, args.log_file)

    mask, original_resolution = _read_group(args.mask)
    shapes = convert_to_shapes(mask, original_resolution, args.threshold)

    _save_shapes(shapes, args.out_file)


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mask',
                        type=str,
                        help='path to the dataset to be converted')
    parser.add_argument(
        '-o',
        dest='out_file',
        type=str,
        help='output file json for the serialized ROIs. Default: STDOUT')
    parser.add_argument(
        '-t',
        dest='threshold',
        type=int,
        required=True,
        help='threshold for generating the ROI. Integer in [0, 100] range')
    parser.add_argument('--log-level',
                        type=str,
                        choices=LOG_LEVELS,
                        default='INFO',
                        help='log level (default=INFO)')
    parser.add_argument('--log-file',
                        type=str,
                        default=None,
                        help='log file (default=stderr)')
    return parser


def _read_group(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    group = zarr.open(path)
    # retrieving the first array
    key = list(group.array_keys())[0]
    mask = group[key]
    mask = np.array(mask)
    resolution = group.attrs['resolution']
    return mask, resolution


def _save_shapes(shapes: Dict, output_path: str):
    if output_path is None:
        print(json.dumps(shapes))
    else:
        with open(output_path, 'w') as ofile:
            json.dump(shapes, ofile)


if __name__ == '__main__':
    main(sys.argv[1:])
