import re
import os
import csv

from config import path_consts

from datetime import datetime


class ResultsLogger(object):
    # Based on: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    @staticmethod
    def _camel_case_to_snake_case(s):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
        return re.sub('(\s+)', '_', s)

    def __init__(self, prefix):
        self._prefix = self._camel_case_to_snake_case(prefix)
        self._base_path = os.path.join(path_consts.RESULTS_FOLDER_PATH, self._prefix,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def get_path_in_dir(self, file):
        return os.path.join(self._base_path, self._camel_case_to_snake_case(file))

    def add_result_to_csv(self, result):
        self._csv.writerow(result)

    def __enter__(self):
        os.makedirs(self._base_path)
        self._csv_output_fd = open(self.get_path_in_dir('cv_results.csv'), 'w')
        self._csv = csv.writer(self._csv_output_fd)
        self._csv.writerow(['path', 'training1', 'training2', 'training3', 'training4', 'training5', 'training_mean',
            'test1', 'test2', 'test3', 'test4', 'test5', 'test_mean'])
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._csv_output_fd.close()