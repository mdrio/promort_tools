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

try:
    import simplejson as json
except ImportError:
    import json

from ..libs.client import ProMortClient, ProMortAuthenticationError

#  from promort_tools.libs.client import ProMortClient, ProMortAuthenticationError

import sys
import requests

PREDICTION_TYPES = ["TISSUE", "TUMOR", "GLEASON"]


class TissueFragmentsImporter(object):
    def __init__(self, host, user, passwd, session_id, logger):
        self.promort_client = ProMortClient(host, user, passwd, session_id)
        self.logger = logger

    def _import_tissue_fragments(self, prediction_id, shapes, provenance_json=None):
        payload = {"label": prediction_id, "shape_json": shapes}
        if provenance_json:
            payload["provenance"] = json.dumps(provenance_json)

        response = self.promort_client.post(api_url="api/predictions/", payload=payload)
        if response.status_code == requests.codes.CREATED:
            self.logger.info("Prediction created")
        elif response.status_code == requests.codes.CONFLICT:
            self.logger.error("A prediction with the same label already exists")
            sys.exit("ERROR: duplicated prediction label")
        elif response.status_code == requests.codes.BAD:
            self.logger.error(
                "ERROR while creating Prediction: {0}".format(response.text)
            )
            sys.exit("ERROR while creating Prediction")

    def run(self, args):
        try:
            self.promort_client.login()
        except ProMortAuthenticationError:
            self.logger.critical("Authentication error, exit")
            sys.exit("Authentication error, exit")

        collection_id = self._create_collection(args.prediction_id)
        self.logger.info("Collection created with id %s", collection_id)

        with open(args.shapes) as f_obj:
            shapes = json.load(f_obj)["shapes"]

        for shape in shapes:
            self.logger.info("add to collection %s shape %s", collection_id, shape)
            self._create_fragment(collection_id, shape)

        self.promort_client.logout()

    def _create_collection(self, prediction_id) -> int:
        response = self.promort_client.post(
            api_url="api/tissue_fragments_collections/",
            payload={"prediction": prediction_id},
        )
        return response.json()["id"]

    def _create_fragment(self, collection_id, shape):
        self.logger.debug("creating shape %s", shape)
        try:
            response = self.promort_client.post(
                api_url=f"api/tissue_fragments_collections/{collection_id}/fragments/",
                json={"shape_json": shape},
            )
            self.logger.debug("response %s", response)
            self.logger.debug("response.text %s", response.text)
            response.raise_for_status()
        except Exception as ex:
            self.logger.error(ex)


help_doc = """
TBD
"""


def implementation(host, user, passwd, session_id, logger, args):
    prediction_importer = TissueFragmentsImporter(
        host, user, passwd, session_id, logger
    )
    prediction_importer.run(args)


def make_parser(parser):
    parser.add_argument(
        "--prediction-id", type=str, required=True, help="prediction id"
    )
    parser.add_argument(
        "shapes", type=str, help="file containing json-serialized shapes"
    )


def register(registration_list):
    registration_list.append(
        ("tissue_fragments_importer", help_doc, make_parser, implementation)
    )
