import pandas as pd

from joblib import parallel_backend
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


class TrainingRunner(object):
    RANDOM_STATE = 10
    VERBOSITY = 0

    def __init__(self, name: str, pipeline: Pipeline, features_data: pd.DataFrame, target_data: pd.DataFrame):
        self._name = name
        self.pipeline = pipeline
        self.X = features_data
        self.y = target_data

    def run_grid_search_cv(self, tuned_parameters, number_of_splits, scoring='neg_mean_squared_error'):
        X_train, _, y_train, _ = train_test_split(self.X, self.y, random_state=self.RANDOM_STATE)
        clf = GridSearchCV(self.pipeline, tuned_parameters, cv=number_of_splits, scoring=scoring, verbose=self.VERBOSITY)
        with parallel_backend('loky'):
            res = clf.fit(X_train, y_train)
        return res

    def run_cross_validation(self, cv, scoring='neg_mean_squared_error', return_estimator=False):
        with parallel_backend('loky'):
            return cross_validate(self.pipeline, self.X, self.y, cv=cv, scoring=scoring,
                return_estimator=return_estimator, return_train_score=True, verbose=self.VERBOSITY)

    def __str__(self):
        return self._name


class SpearmanCorrelationPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 50):
        pipeline = Pipeline([('feature_selection_k_best', SelectKBest(spearmanr, k)),
                             ('training_model', MultiOutputRegressor(training_model))])
        super().__init__(name, pipeline, features_data, target_data)


class ModelFeatureSlectionPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, feature_selection_model, features_data: pd.DataFrame, target_data: pd.DataFrame, max_features: int = 50):
        pipeline = Pipeline([('feature_selection_k_best', SelectFromModel(estimator=feature_selection_model, max_features=max_features)),
                             ('training_model', MultiOutputRegressor(training_model))])
        super().__init__(name, pipeline, features_data, target_data)


class PCAPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, n_components: float = 6):
        pipeline = Pipeline([('pca', PCA(n_components=n_components)),
                             ('training_model', training_model)])
        super().__init__(name, pipeline, features_data, target_data)


class RawPipelineRunner(TrainingRunner):
    def __init__(self, name: str, model, features_data: pd.DataFrame, target_data: pd.DataFrame):
        pipeline = Pipeline([('model', model)])
        super().__init__(name, pipeline, features_data, target_data)


class SemisupervisedPipelineRunner(TrainingRunner):
    class SemisupervisedModelWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, model, unsupervised_data: pd.DataFrame):
            self._unsupervised_data = unsupervised_data
            self._model = model

        def _add_unsupervised_data(self, X, y):
            self._model.fit(X, y)
            y_unsupervised_data = self._model.predict(self._unsupervised_data)
            return X.append(self._unsupervised_data), y.append(y_unsupervised_data)

        def fit(self, X, y):
            X_with_unsupervised_data, y_with_unsupervised_data = self._add_unsupervised_data(X, y)
            self._model.fit(X_with_unsupervised_data, y_with_unsupervised_data)

            return self

        def predict(self, X):
            return self._model.predict(X)

    def __init__(self, name: str, model, supervised_features_data: pd.DataFrame, target_data: pd.DataFrame, un_supervised_features_data: pd.DataFrame):
        pipeline = Pipeline([('model', self.SemisupervisedModelWrapper(model, un_supervised_features_data))])
        super().__init__(name, pipeline, supervised_features_data, target_data)

