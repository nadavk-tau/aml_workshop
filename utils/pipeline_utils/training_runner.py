import pandas as pd

from typing import Callable
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


class TrainingRunner:

    def __init__(self, pipeline: Pipeline, features_data: pd.DataFrame, target_data: pd.DataFrame):
        self._pipeline = pipeline
        self.X = features_data
        self.y = target_data


    def run_cross_validation(self, number_of_splits: int, number_of_repeats: int, accuracy_function: Callable = mean_squared_error) -> pd.DataFrame:
        rkf = RepeatedKFold(n_splits=number_of_splits, n_repeats=number_of_repeats)

        error_measures = {column: [] for column in self.y.columns}
        for current_target in self.y.columns:
            current_y = self.y[current_target]

            for train_index, test_index in rkf.split(self.X):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = current_y.iloc[train_index], current_y.iloc[test_index]

                clf = self._pipeline.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                current_mse = accuracy_function(y_test, y_pred)
                error_measures[current_target].append(current_mse)


        return pd.DataFrame(error_measures)


class SpearmanCorrelationPipelineRunner(TrainingRunner):

    def __init__(self, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 50):
        pipeline = Pipeline([('feature_selection_k_best', SelectKBest(spearmanr, k)),
                             ('training_mode', training_model)])

        super().__init__(pipeline, features_data, target_data)


class ModelFeatureSlectionPipelineRunner(TrainingRunner):

    def __init__(self, training_model, feature_selection_model, features_data: pd.DataFrame, target_data: pd.DataFrame, max_features: int = 50):
        pipeline = Pipeline([('feature_selection_k_best', SelectFromModel(estimator=feature_selection_model, max_features=max_features)),
                             ('training_mode', training_model)])

        super().__init__(pipeline, features_data, target_data)