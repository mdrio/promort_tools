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
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import zarr
from shapely.geometry import Polygon

from promort_tools.libs.utils.logger import LOG_LEVELS, get_logger

LOGGER = logging.getLogger()

COORDS = Tuple[float, float]


def convert_to_shapes(
    mask: np.ndarray,
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

    def _build_slide_json(cores):
        slide_shapes = [
            {
                "coordinates": c.get_coordinates(),
                "length": c.get_length(),
                "area": c.get_area(),
            }
            for c in cores
        ]
        return {"shapes": slide_shapes}

    def _scale_cores(scaler):
        return [scaler.scale(core) for core in cores]

    _apply_threshold(mask, threshold)

    cores = _get_cores(mask)
    cores = _scale_cores(scaler)
    cores = _filter_cores(cores, mask.size)
    return _build_slide_json(cores)


class Shape:
    def __init__(self, segments):
        self.polygon = Polygon(segments)

    def __str__(self):
        return str(self.polygon)

    def get_bounds(self):
        bounds = self.polygon.bounds
        try:
            return {
                "x_min": bounds[0],
                "y_min": bounds[1],
                "x_max": bounds[2],
                "y_max": bounds[3],
            }
        except IndexError:
            raise InvalidPolygonError()

    def get_coordinates(self):
        return list(self.polygon.exterior.coords)

    def get_area(self):
        return self.polygon.area

    def get_length(self):
        return self.polygon.length

    #  def get_full_mask(self, scale_level=0, tolerance=0):
    #      if scale_level != 0:
    #          polygon = self._rescale_polygon(scale_level)
    #          scale_factor = pow(2, scale_level)
    #      else:
    #          polygon = self.polygon
    #          scale_factor = 1
    #      if tolerance > 0:
    #          polygon = polygon.simplify(tolerance, preserve_topology=False)
    #      bounds = self.get_bounds()
    #      box_height = int((bounds["y_max"] - bounds["y_min"]) * scale_factor)
    #      box_width = int((bounds["x_max"] - bounds["x_min"]) * scale_factor)
    #      mask = np.zeros((box_height, box_width), dtype=np.uint8)
    #      polygon_path = polygon.exterior.coords[:]
    #      polygon_path = [
    #          (
    #              int(x - bounds["x_min"] * scale_factor),
    #              int(y - bounds["y_min"] * scale_factor),
    #          )
    #          for x, y in polygon_path
    #      ]
    #      cv2.fillPoly(
    #          mask,
    #          np.array(
    #              [
    #                  polygon_path,
    #              ]
    #          ),
    #          1,
    #      )
    #      return mask


class Scaler(abc.ABC):
    @abc.abstractmethod
    def scale(self, shape: Shape) -> Shape:
        ...


@dataclass
class BasicScaler(Scaler):
    init_resolution: Tuple[int, int]
    dest_resolution: Tuple[int, int]

    def scale(self, shape: Shape) -> Shape:
        if self.init_resolution == self.dest_resolution:
            return shape

        polygon = shape.polygon
        points = np.array(list(polygon.exterior.coords))
        points = points + 0.5

        norm_points = points / np.array(self.init_resolution)
        new_points = norm_points * np.array(self.dest_resolution)
        return Shape(new_points)


def main(argv):
    parser = _make_parser()
    args = parser.parse_args(argv)

    global LOGGER
    LOGGER = get_logger(args.log_level, args.log_file)

    mask, original_resolution, round_to_0_100 = _read_group(args.mask)
    threshold = round(args.threshold * 100) if round_to_0_100 else args.threshold

    scaler = BasicScaler(mask.shape, original_resolution)
    shapes = convert_to_shapes(mask, threshold, scaler)

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
