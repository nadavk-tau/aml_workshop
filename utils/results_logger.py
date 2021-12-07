import re
import os
import csv

from config import path_consts

from datetime import datetime


class ResultsLogger(object):
    COMMON_RESULT_COLUMNS = ['path', 'training1', 'training2', 'training3', 'training4', 'training5', 'training_mean',
            'test1', 'test2', 'test3', 'test4', 'test5', 'test_mean']


    # Based on: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    @staticmethod
    def _camel_case_to_snake_case(s):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
        return re.sub('(\s+)', '_', s)

    def __init__(self, prefix, results_columns=None):
        self._prefix = self._camel_case_to_snake_case(prefix)
        self._base_path = os.path.join(path_consts.RESULTS_FOLDER_PATH, self._prefix,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.results_columns = results_columns if results_columns else self.COMMON_RESULT_COLUMNS

    def get_path_in_dir(self, file):
        return os.path.join(self._base_path, self._camel_case_to_snake_case(file))

    def add_result_to_cv_results_csv(self, result):
        self._cv_results_csv.writerow(result)

    def _write_file(self, output_file_name, readable_name, callback, **kwargs):
        full_output_file_name = self.get_path_in_dir(output_file_name)
        print(f"- Writing {readable_name} to '{full_output_file_name}'... ", end='')
        callback(full_output_file_name, **kwargs)
        print('Done.')
        return full_output_file_name

    def save_csv(self, output_file_name, readable_name, dataframe, **kwargs):
        return self._write_file(output_file_name, readable_name, dataframe.to_csv, **kwargs)

    def save_figure(self, output_file_name, readable_name, fig, **kwargs):
        return self._write_file(output_file_name, readable_name, fig.savefig, **kwargs)

    def __enter__(self):
        os.makedirs(self._base_path)
        self._cv_results_csv_fd = open(self.get_path_in_dir('cv_results.csv'), 'w')
        self._cv_results_csv = csv.writer(self._cv_results_csv_fd)
        self._cv_results_csv.writerow(self.results_columns)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._cv_results_csv_fd.close()
