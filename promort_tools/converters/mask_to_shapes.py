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
import abc
import argparse
import json
import logging
import sys
from math import sqrt
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import zarr
from shapely.geometry import Point, Polygon

from promort_tools.libs.utils.logger import LOG_LEVELS, get_logger

LOGGER = logging.getLogger()


COORDS = Tuple[float, float]


def convert_to_shapes(
    mask: np.ndarray,
    original_resolution: Tuple[int, int],
    threshold: int,
    scaler: "Scaler",
):
    def _apply_threshold(mask: np.ndarray, threshold: int) -> np.ndarray:
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1

    def _get_cores(mask):
        contours, _ = cv2.findContours(
            mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        LOGGER.debug("contours %s", contours)
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
        scale_factor = sqrt(
            (slide_resolution[0] * slide_resolution[1])
            / (mask_resolution[0] * mask_resolution[1])
        )
        LOGGER.info("Scale factor is %r", scale_factor)
        return scale_factor

    def _build_slide_json(cores, scale_factor):
        slide_shapes = [scaler.scale(core, scale_factor).info() for core in cores]
        return {"shapes": slide_shapes}

    _apply_threshold(mask, threshold)

    cores = _filter_cores(_get_cores(mask), mask.size)
    #  grouped_cores = self._group_nearest_cores(cores, mask.shape[0])
    scale_factor = _get_scale_factor(original_resolution, mask.shape)
    return _build_slide_json(cores, scale_factor)


class Shape:
    def __init__(self, segments):
        self._polygon = Polygon(segments)
        self._area = self._polygon.area

    def __str__(self):
        return str(self._polygon)

    def get_bounds(self):
        bounds = self._polygon.bounds
        try:
            return {
                "x_min": bounds[0],
                "y_min": bounds[1],
                "x_max": bounds[2],
                "y_max": bounds[3],
            }
        except IndexError:
            raise InvalidPolygonError()

    def get_coordinates(
        self,
    ) -> List[COORDS]:
        return list(self._polygon.exterior.coords)

    def get_area(self) -> float:
        return self._area

    def get_length(self) -> float:
        #  polygon_path = np.array(self._polygon.exterior.coords[:])
        #  _, radius = cv2.minEnclosingCircle(polygon_path.astype(int))
        #  return radius * 2

        # from https://gis.stackexchange.com/questions/295874/getting-polygon-breadth-in-shapely
        box = self._polygon.minimum_rotated_rectangle
        x, y = box.exterior.coords.xy

        edge_length = (
            Point(x[0], y[0]).distance(Point(x[1], y[1])),
            Point(x[1], y[1]).distance(Point(x[2], y[2])),
        )
        length = max(edge_length)
        return length

    def info(self):
        return {
            "coordinates": self.get_coordinates(),
            "length": self.get_length(),
            "area": self.get_area(),
        }


class Scaler(abc.ABC):
    @abc.abstractmethod
    def scale(self, shape: Shape, factor: float) -> Shape:
        ...


class BasicScaler(Scaler):
    def scale(self, shape: Shape, factor) -> Shape:
        points = np.array(shape.get_coordinates())
        points = points + 0.5
        return Shape(Polygon(points * factor))


def main(argv):
    parser = _make_parser()
    args = parser.parse_args(argv)

    global LOGGER
    LOGGER = get_logger(args.log_level, args.log_file)

    mask, original_resolution, round_to_0_100 = _read_group(args.mask)
    threshold = round(args.threshold * 100) if round_to_0_100 else args.threshold

    scaler = BasicScaler()
    shapes = convert_to_shapes(mask, original_resolution, threshold, scaler)

    _save_shapes(shapes, args.out_file)


def _get_scale_func(func_name: str) -> Callable:
    return globals()[f"{func_name}_scale"]


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=str, help="path to the dataset to be converted")
    parser.add_argument(
        "-o",
        dest="out_file",
        type=str,
        help="output file json for the serialized ROIs. Default: STDOUT",
    )
    parser.add_argument(
        "-t",
        dest="threshold",
        type=float,
        required=True,
        help="threshold for generating the ROI. Float in range [0, 1].",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=LOG_LEVELS,
        default="INFO",
        help="log level (default=INFO)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="log file (default=stderr)"
    )

    scale_funcs = ("shapely", "fit", "pyclipper")
    parser.add_argument(
        "--scale-func",
        dest="scale_func",
        type=str,
        choices=scale_funcs,
        default=scale_funcs[0],
        help="log file (default=stderr)",
    )
    return parser


def _read_group(path: str) -> Tuple[np.ndarray, Tuple[int, int, bool]]:
    group = zarr.open(path)
    # retrieving the first array
    key = list(group.array_keys())[0]
    mask = group[key]
    round_to_0_100 = mask.attrs["round_to_0_100"]
    mask = np.array(mask)
    resolution = group.attrs["resolution"]
    return mask, resolution, round_to_0_100


def _save_shapes(shapes: Dict, output_path: str):
    if output_path is None:
        print(json.dumps(shapes))
    else:
        with open(output_path, "w") as ofile:
            json.dump(shapes, ofile)


class InvalidPolygonError(Exception):
    ...


if __name__ == "__main__":
    main(sys.argv[1:])
