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

import argparse, sys, os
import zarr
import tiledb
import numpy as np
from math import ceil

from promort_tools.libs.utils.logger import get_logger, LOG_LEVELS


class ZarrToTileDBConverter(object):

    def __init__(self, logger):
        self.logger = logger

    def _get_array_shape(self, zarr_dataset):
        shapes = set([arr[1].shape for arr in zarr_dataset.arrays()])
        if len(shapes) == 1:
            return shapes.pop()
        else:
            self.logger.error('Multiple shapes in zarr dataset arrays, cannot convert to tiledb')
            sys.exit('Multiple shapes in zarr arrays')

    def _get_array_attributes(self, zarr_dataset):
        return [(a[0], a[1].dtype) for a in zarr_dataset.arrays()]

    def _get_tiledb_path(self, zarr_dataset, out_folder):
        return os.path.join(
            out_folder,
            '{0}.tiledb'.format(os.path.splitext(os.path.basename(zarr_dataset.attrs['filename']))[0])
        )

    def _init_tiledb_dataset(self, dataset_path, dataset_shape, zarr_attributes):
        rows = tiledb.Dim(name='rows', domain=(0, dataset_shape[0]-1), tile=4, dtype=np.uint16)
        columns = tiledb.Dim(name='columns', domain=(0, dataset_shape[1]-1), tile=4, dtype=np.uint16)
        domain = tiledb.Domain(rows, columns)
        attributes = list()
        for a in zarr_attributes:
            attributes.append(tiledb.Attr(a[0], dtype=a[1]))
        schema = tiledb.ArraySchema(domain=domain, sparse=False, attrs=attributes)
        tiledb.DenseArray.create(dataset_path, schema)

    def _zarr_to_tiledb(self, zarr_dataset, tiledb_dataset_path, slide_resolution):
        tiledb_data = dict()
        tiledb_meta = {
            'original_width': slide_resolution[0],
            'original_height': slide_resolution[1]
        }
        for arr_label, arr_data in zarr_dataset.arrays():
            tiledb_data[arr_label] = arr_data[:]
            tiledb_meta.update(
                {
                    '{0}.dzi_sampling_level'.format(arr_label): ceil(arr_data.attrs['dzi_sampling_level']),
                    '{0}.tile_size'.format(arr_label): arr_data.attrs['tile_size'],
                    '{0}.rows'.format(arr_label): arr_data.shape[1],
                    '{0}.columns'.format(arr_label): arr_data.shape[0]
                }
            )
        with tiledb.open(tiledb_dataset_path, 'w') as A:
            A[:] = tiledb_data
            for k, v in tiledb_meta.items():
                A.meta[k] = v

    def run(self, zarr_dataset, out_folder):
        z = zarr.open(zarr_dataset)
        try:
            slide_res = z.attrs['resolution']
        except KeyError as ke:
            self.logger.error('Missing key {0} in zarr attributes, exit'.format(ke))
            sys.exit('Missing key {0}'.format(ke))
        dset_shape = self._get_array_shape(z)
        tiledb_dataset_path = self._get_tiledb_path(z, out_folder)
        self.logger.info('TileDB dataset path: {0}'.format(tiledb_dataset_path))
        attributes = self._get_array_attributes(z)
        self._init_tiledb_dataset(tiledb_dataset_path, dset_shape, attributes)
        self._zarr_to_tiledb(z, tiledb_dataset_path, slide_res)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr-dataset', type=str, required=True,
                        help='path to the ZARR dataset to be converted')
    parser.add_argument('--out-folder', type=str, required=True,
                        help='output folder for TileDB dataset')
    parser.add_argument('--log-level', type=str, choices=LOG_LEVELS,
                        default='INFO', help='log level (default=INFO)')
    parser.add_argument('--log-file', type=str, default=None, help='log file (default=stderr)')
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    logger = get_logger(args.log_level, args.log_file)
    app = ZarrToTileDBConverter(logger)
    app.run(args.zarr_dataset, args.out_folder)


if __name__ == '__main__':
    main(sys.argv[1:])
