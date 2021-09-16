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
import os
import sys
from math import log, sqrt
from random import randint

import cv2
import numpy as np
import zarr
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Point, Polygon

from promort_tools.libs.utils.logger import LOG_LEVELS, get_logger


class MaskToROIConverter:
    def __init__(self, logger):
        self.logger = logger

    def run(self, mask_path, threshold, output_path=None):

        mask, original_resolution = self._load_mask(mask_path)

        self._apply_threshold(mask, threshold)

        cores = self._filter_cores(self._get_cores(mask), mask.size)
        #  grouped_cores = self._group_nearest_cores(cores, mask.shape[0])
        scale_factor = self._get_scale_factor(original_resolution, mask.shape)
        slide_json = self._build_slide_json(cores, scale_factor)

        output_path = output_path or f'{os.path.splitext(mask_path)[0]}.json'
        self._save(slide_json, output_path)

    @staticmethod
    def _apply_threshold(mask: np.ndarray, threshold: int) -> np.ndarray:
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1

    @staticmethod
    def _save(slide_json, output_path):
        with open(output_path, 'w') as ofile:
            ofile.write(json.dumps(slide_json))

    def _load_mask(self, mask_path):
        group = zarr.open(mask_path)
        # retrieving the first array
        key = list(group.array_keys())[0]
        mask = group[key]
        mask = np.array(mask)
        resolution = group.attrs['resolution']
        return mask, resolution

    def _get_cores(self, mask):
        contours, _ = cv2.findContours(mask,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(None, [self._contour_to_shape(c) for c in contours])
        return contours

    def _contour_to_shape(self, contour):
        normalize_contour = []
        for x in contour:
            normalize_contour.append(tuple(x[0]))
        try:
            return Shape(normalize_contour)
        except ValueError:
            return None

    def _filter_cores(self, cores, slide_area, core_min_area=0.02):
        accepted_cores = []
        for core in cores:
            if (core.get_area() * 100 / slide_area) >= core_min_area:
                accepted_cores.append(core)
        return accepted_cores

    def _get_scale_factor(self, slide_resolution, mask_resolution):
        scale_factor = sqrt((slide_resolution[0] * slide_resolution[1]) /
                            (mask_resolution[0] * mask_resolution[1]))
        self.logger.info('Scale factor is %r', scale_factor)
        return scale_factor

    def _build_slide_json(self, cores, scale_factor):
        scale_factor = log(scale_factor, 2)
        slide_shapes = [{
            'coordinates': c.get_coordinates(scale_factor),
            'length': c.get_length(scale_factor),
            'area': c.get_area(scale_factor)
        } for c in cores]
        return slide_shapes

    def _get_slice(self, cores_group):
        x_min = min([c.get_bounds()['x_min'] for c in cores_group])
        y_min = min([c.get_bounds()['y_min'] for c in cores_group])
        x_max = max([c.get_bounds()['x_max'] for c in cores_group])
        y_max = max([c.get_bounds()['y_max'] for c in cores_group])
        return Shape([(x_min, y_min), (x_max, y_min), (x_max, y_max),
                      (x_min, y_max)])

    def _group_nearest_cores(self, cores, slide_height, height_tolerance=0.01):
        cores_map, sorted_y_coords = self._get_sorted_cores_map(cores)
        cores_groups = []
        tolerance = slide_height * height_tolerance
        current_group = cores_map[sorted_y_coords[0]]
        for i, yc in enumerate(sorted_y_coords[1:]):
            if yc[0] <= sorted_y_coords[i][1] + tolerance:
                current_group.extend(cores_map[yc])
            else:
                cores_groups.append(current_group)
                current_group = cores_map[yc]
        cores_groups.append(current_group)
        return cores_groups

    def _get_sorted_cores_map(self, cores):
        cores_map = dict()
        for c in cores:
            bounds = c.get_bounds()
            cores_map.setdefault((bounds['y_min'], bounds['y_max']),
                                 []).append(c)
        sorted_y_coords = cores_map.keys()
        sorted_y_coords.sort(key=lambda x: x[0])
        return cores_map, sorted_y_coords


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

    def get_bounding_box(self, x_min=None, y_min=None, x_max=None, y_max=None):
        p1, p2, p3, p4 = self.get_bounding_box_points(x_min, y_min, x_max,
                                                      y_max)
        return self._box_to_polygon({
            'up_left': p1,
            'up_right': p2,
            'down_right': p3,
            'down_left': p4
        })

    def get_bounding_box_points(self,
                                x_min=None,
                                y_min=None,
                                x_max=None,
                                y_max=None):
        bounds = self.get_bounds()
        xm = x_min if not x_min is None else bounds['x_min']
        xM = x_max if not x_max is None else bounds['x_max']
        ym = y_min if not y_min is None else bounds['y_min']
        yM = y_max if not y_max is None else bounds['y_max']
        return [(xm, ym), (xM, ym), (xM, yM), (xm, yM)]

    def get_random_point(self):
        bounds = self.get_bounds()
        point = Point(randint(int(bounds['x_min']), int(bounds['x_max'])),
                      randint(int(bounds['y_min']), int(bounds['y_max'])))
        while not self.polygon.contains(point):
            point = Point(randint(int(bounds['x_min']), int(bounds['x_max'])),
                          randint(int(bounds['y_min']), int(bounds['y_max'])))
        return point

    def get_random_points(self, points_count):
        points = [self.get_random_point() for _ in range(points_count)]
        return points

    def _box_to_polygon(self, box):
        return Polygon([
            box['down_left'], box['down_right'], box['up_right'],
            box['up_left']
        ])

    def _rescale_polygon(self, scale_level):
        scaling = pow(2, scale_level)
        return scale(self.polygon, xfact=scaling, yfact=scaling, origin=(0, 0))

    def get_intersection_mask(self, box, scale_level=0, tolerance=0):
        if scale_level != 0:
            polygon = self._rescale_polygon(scale_level)
        else:
            polygon = self.polygon
        if tolerance > 0:
            polygon = polygon.simplify(tolerance, preserve_topology=False)
        box_polygon = self._box_to_polygon(box)
        box_height = int(box['down_left'][1] - box['up_left'][1])
        box_width = int(box['down_right'][0] - box['down_left'][0])
        if not polygon.intersects(box_polygon):
            return np.zeros((box_width, box_height), dtype=np.uint8)
        else:
            if polygon.contains(box_polygon):
                return np.ones((box_width, box_height), dtype=np.uint8)
            else:
                mask = np.zeros((box_width, box_height), dtype=np.uint8)
                intersection = polygon.intersection(box_polygon)
                if type(intersection) is MultiPolygon:
                    intersection_paths = list(intersection)
                else:
                    intersection_paths = [intersection]
                for path in intersection_paths:
                    ipath = path.exterior.coords[:]
                    ipath = [(int(x - box['up_left'][0]),
                              int(y - box['up_left'][1])) for x, y in ipath]
                    cv2.fillPoly(mask, np.array([
                        ipath,
                    ]), 1)
                return mask

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

    def get_difference_mask(self, box, scale_level=0, tolerance=0):
        return 1 - self.get_intersection_mask(box, scale_level, tolerance)


class InvalidPolygonError(Exception):
    ...


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mask',
                        type=str,
                        help='path to the dataset to be converted')
    parser.add_argument(
        '--out-file',
        type=str,
        help=
        'output file json for the serialized ROIs. Default: mask path with json extension'
    )
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


def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)
    logger = get_logger(args.log_level, args.log_file)
    app = MaskToROIConverter(logger)
    app.run(args.mask, args.threshold, args.out_file)


if __name__ == '__main__':
    main(sys.argv[1:])
