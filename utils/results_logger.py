import re
import os

from config import path_consts

from datetime import datetime


class ResultsDir(object):
    # Based on: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    @staticmethod
    def _camel_case_to_snake_case(s):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
        return re.sub('(\s+)', '_', s)

    def __init__(self, prefix):
        self._prefix = self._camel_case_to_snake_case(prefix)
        self._path = os.path.join(path_consts.RESULTS_FOLDER_PATH, self._prefix,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def get_path_in_dir(self, file):
        return os.path.join(self._path, self._camel_case_to_snake_case(file))

    def __enter__(self):
        os.makedirs(self._path)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass
