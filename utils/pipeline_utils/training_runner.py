import pandas as pd

from joblib import parallel_backend
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


class TrainingRunner(object):
    RANDOM_STATE = 10
    VERBOSITY = 3

    def __init__(self, pipeline: Pipeline, features_data: pd.DataFrame, target_data: pd.DataFrame):
        self._pipeline = pipeline
        self.X = features_data
        self.y = target_data

    def run_grid_search_cv(self, tuned_parameters, number_of_splits, scoring='neg_mean_squared_error'):
        X_train, _, y_train, _ = train_test_split(self.X, self.y, random_state=self.RANDOM_STATE) 
        clf = GridSearchCV(self._pipeline, tuned_parameters, cv=number_of_splits, scoring=scoring, verbose=self.VERBOSITY)
        with parallel_backend('loky'):
            res = clf.fit(X_train, y_train)
        return res

    def run_cross_validation(self, number_of_splits: int, scoring: str) -> pd.DataFrame:
        with parallel_backend('loky'):
            return cross_validate(self._pipeline, self.X, self.y, cv=number_of_splits, scoring=scoring, verbose=self.VERBOSITY) 


class SpearmanCorrelationPipelineRunner(TrainingRunner):
    def __init__(self, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 50):
        pipeline = Pipeline([('feature_selection_k_best', SelectKBest(spearmanr, k)),
                             ('training_model', MultiOutputRegressor(training_model))])
        super().__init__(pipeline, features_data, target_data)


class ModelFeatureSlectionPipelineRunner(TrainingRunner):
    def __init__(self, training_model, feature_selection_model, features_data: pd.DataFrame, target_data: pd.DataFrame, max_features: int = 50):
        pipeline = Pipeline([('feature_selection_k_best', SelectFromModel(estimator=feature_selection_model, max_features=max_features)),
                             ('training_model', MultiOutputRegressor(training_model))])
        super().__init__(pipeline, features_data, target_data)


class LassoPipelineRunner(TrainingRunner):
    def __init__(self, features_data: pd.DataFrame, target_data: pd.DataFrame, alpha: float = 1.0):
        pipeline = Pipeline([('training_mode', MultiTaskLasso(random_state=10, max_iter=10000, alpha=alpha))])
        super().__init__(pipeline, features_data, target_data)
