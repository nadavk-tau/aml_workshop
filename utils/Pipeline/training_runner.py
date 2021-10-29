from typing import Callable
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error


class TrainingRunner:

    def __init__(self, model, features_data: pd.DataFrame, target_data: pd.DataFrame):
        self._model = model

        # NOTE: sklearn is expecting to get the features as columns
        # TODO: move this to the data parsing stage
        self.X = features_data.T
        self.y = target_data.T


    def run_cross_validation(self, number_of_splits: int, number_of_repeats: int, accuracy_function: Callable = mean_squared_error) -> pd.DataFrame:
        rkf = RepeatedKFold(n_splits=number_of_splits, n_repeats=number_of_repeats)

        mse_results = {column: [] for column in self.y.columns}
        for current_target in self.y.columns:
            current_y = self.y[current_target]

            for train_index, test_index in rkf.split(self.X):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = current_y.iloc[train_index], current_y.iloc[test_index]

                clf = self._model.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                current_mse = accuracy_function(y_test, y_pred)
                mse_results[current_target].append(current_mse)


        return pd.DataFrame(mse_results)