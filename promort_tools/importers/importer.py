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

import argparse, logging, sys
from importlib import import_module

SUBMODULES_NAMES = [
    'slides_importer'
]

SUBMODULES = [import_module('%s.%s' % ('promort_tools.importers', n)) for n in SUBMODULES_NAMES]

LOG_FORMAT = '%(asctime)s|%(levelname)-8s|%(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


class ProMortImporter(object):

    def __init__(self):
        self.supported_modules = []
        for m in SUBMODULES:
            m.register(self.supported_modules)

    def make_parser(self):
        parser = argparse.ArgumentParser(description='ProMort data importer')
        parser.add_argument('--host', type=str, required=True, help='ProMort host')
        parser.add_argument('--user', type=str, required=True, help='ProMort user')
        parser.add_argument('--passwd', type=str, required=True, help='ProMost password')
        parser.add_argument('--session-id', type=str, default='promort_sessionid',
                            help='ProMort session cookie name')
        parser.add_argument('--log-level', type=str, choices=LOG_LEVELS,
                            default='INFO', help='logging level (default=INFO')
        parser.add_argument('--log-file', type=str, default=None, help='log file (default=stderr)')
        subparsers = parser.add_subparsers()
        for k, h, addp, impl in self.supported_modules:
            subparser = subparsers.add_parser(k, help=h)
            addp(subparser)
            subparser.set_defaults(func=impl)
        return parser

    def get_logger(self, log_level, log_file, mode='a'):
        logger = logging.getLogger('odin')
        if not isinstance(log_level, int):
            try:
                log_level = getattr(logging, log_level)
            except AttributeError:
                raise ValueError('Unsupported literal log level: %s' % log_level)
        logger.setLevel(log_level)
        logger.handlers = []
        if log_file:
            handler = logging.FileHandler(log_file, mode=mode)
        else:
            handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


def main(argv=None):
    app = ProMortImporter()
    parser = app.make_parser()
    args = parser.parse_args(argv)
    logger = app.get_logger(args.log_level, args.log_file)
    try:
        args.func(args.host, args.user, args.passwd, args.session_id, logger, args)
    except argparse.ArgumentError as arg_err:
        logger.critical(arg_err)
        sys.exit(arg_err)


if __name__ == '__main__':
    main(sys.argv[1:])
