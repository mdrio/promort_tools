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

from ..libs.client import ProMortClient
from ..libs.client import ProMortAuthenticationError

from argparse import ArgumentError
import sys, requests
from urllib.parse import urljoin
from functools import reduce


class SlideImporter(object):

    def __init__(self, host, user, passwd, logger):
        self.promort_client = ProMortClient(host, user, passwd, 'promort_sessionid')
        self.logger = logger

    def _get_case_label(self, slide_label):
        return slide_label.split('-')[0]

    def _import_case(self, case_label):
        response = self.promort_client.post(
            api_url='api/cases/',
            payload={'id': case_label}
        )
        if response.status_code == requests.codes.CREATED:
            self.logger.info('Case created')
        elif response.status_code == requests.codes.CONFLICT:
            self.logger.info('Case already exist')
        elif response.status_code == requests.codes.BAD:
            self.logger.error('ERROR while creating Case: {0}'.format(response.text))
            sys.exit('ERROR while creating Case')

    def _import_slide(self, slide_label, case_label, omero_id=None, mirax_file=False, omero_host=None):
        if mirax_file:
            file_type = 'MIRAX'
        else:
            file_type = 'OMERO_IMG'
        response = self.promort_client.post(
            api_url='api/slides/',
            payload={'id': slide_label, 'case': case_label, 'omero_id': omero_id, 'image_type': file_type}
        )
        if response.status_code == requests.codes.CREATED:
            self.logger.info('Slide created')
            if omero_id is not None and omero_host is not None:
                self._update_slide(slide_label, omero_id, mirax_file, omero_host)
        elif response.status_code == requests.codes.CONFLICT:
            self.logger.error('A slide with the same ID already exists')
            sys.exit('ERROR: duplicated slide')
        elif response.status_code == requests.codes.BAD:
            self.logger.error('ERROR while creating Slide: {0}'.format(response.text))
            sys.exit('ERROR while creating Slide')

    def _update_slide(self, slide_label, omero_id, mirax_file, omero_host):
        if mirax_file:
            join_items = (omero_host, 'ome_seadragon/mirax/deepzoom/get/', '{0}_metadata.json'.format(slide_label))
        else:
            join_items = (omero_host, 'ome_seadragon/deepzoom/get/', '{0}_metadata.json'.format(omero_id))
        ome_url = reduce(urljoin, join_items)
        response = requests.get(ome_url)
        if response.status_code == requests.codes.OK:
            slide_mpp = response.json()['image_mpp']
            response = self.promort_client.put(
                api_url='api/slides/{0}/'.format(slide_label),
                payload={'image_microns_per_pixel': slide_mpp}
            )
            self.logger.info('Slide updated')

    def run(self, args):
        if args.case_label is None and not args.extract_case:
            raise ArgumentError(args.case_label,
                                message='ERROR! Must specify a case label or enable the extract-case flag')
        if args.case_label is not None:
            if args.extract_case:
                self.logger.info('Using label passed through CLI, ignoring the extract-case flag')
            case_label = args.case_label
        else:
            case_label = self._get_case_label(args.slide_label)
        try:
            self.promort_client.login()
        except ProMortAuthenticationError:
            self.logger.critical('Authentication error, exit')
            sys.exit('Authentication error, exit')
        self._import_case(case_label)
        self._import_slide(args.slide_label, case_label, args.omero_id, args.mirax, args.omero_host)
        self.logger.info('Import job completed')
        self.promort_client.logout()


help_doc = """
TBD
"""


def implementation(host, user, passwd, logger, args):
    slide_importer = SlideImporter(host, user, passwd, logger)
    slide_importer.run(args)


def make_parser(parser):
    parser.add_argument('--slide-label', type=str, required=True, help='slide label')
    parser.add_argument('--case-label', type=str, required=False, help='case label')
    parser.add_argument('--omero-id', type=int,
                        help='OMERO ID, only required if the slide was previously uploaded to an OMERO server')
    parser.add_argument('--omero-host', type=str,
                        help='OMERO host used to retrieve slide details (if omero-id was specified)')
    parser.add_argument('--mirax', action='store_true', help='slide is a 3DHISTECH MIRAX')
    parser.add_argument('--extract-case', action='store_true', help='extract case ID from slide label')


def register(registration_list):
    registration_list.append(('slides_importer', help_doc, make_parser, implementation))
