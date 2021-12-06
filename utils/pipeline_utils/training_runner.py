import pandas as pd
import numpy as np

from utils import mutation_matrix_utils

from joblib import parallel_backend
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression, mutual_info_regression, RFE
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, mannwhitneyu, f_oneway
from sklearn.metrics import mean_squared_error, r2_score


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
    
    @staticmethod
    def _metric_matrix_to_dataframe(metric_matrix, num_folds, column_names):
        metric_dataframe = pd.DataFrame(metric_matrix, index=pd.Index(range(num_folds), name='fold'),
            columns=column_names)
        metric_dataframe.loc['mean'] = metric_dataframe.mean()
        return metric_dataframe

    def run_cross_validation_and_get_estimated_results(self, cv, scoring='neg_mean_squared_error'):
        cv_results = self.run_cross_validation(cv, scoring=scoring, return_estimator=True)
        estimated_results = pd.DataFrame()
        mse_folds = []
        r2_folds = []
        for i, (_, test_indexes) in enumerate(cv):
            test_patients = self.X.iloc[test_indexes]
            results_true = self.y.iloc[test_indexes]
            results_pred = pd.DataFrame(cv_results['estimator'][i].predict(test_patients),
                index=test_patients.index, columns=results_true.columns)
            estimated_results = estimated_results.append(results_pred)
            mse_folds.append(mean_squared_error(results_true, results_pred, multioutput='raw_values'))
            r2_folds.append(r2_score(results_true, results_pred, multioutput='raw_values'))
        cv_results['estimated_results'] = estimated_results
        cv_results['mse_matrix'] = self._metric_matrix_to_dataframe(np.row_stack(mse_folds), len(cv), self.y.columns)
        cv_results['r2_matrix'] = self._metric_matrix_to_dataframe(np.row_stack(r2_folds), len(cv), self.y.columns)
        return cv_results

    def __str__(self):
        return self._name


class PcaPartial(PCA):
    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0, iterated_power='auto', random_state=None, extra_x=None):
        self.extra_x = extra_x
        super().__init__(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)

    def _drop_extra(self, X):
        return X[:-len(self.extra_x), :]

    def fit(self, X, y=None):
        res = super().fit(X.append(self.extra_x), y=None) # Drop y
        return self._drop_extra(res)

    def fit_transform(self, X, y=None):
        res = super().fit_transform(X.append(self.extra_x), y=None) # Drop y
        return self._drop_extra(res)

    def transform(self, X):
        return super().transform(X)


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

class RFEFeatureSlectionPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, feature_selection_model, features_data: pd.DataFrame, target_data: pd.DataFrame, n_features_to_select=0.3):
        pipeline = Pipeline([('feature_selection_k_best', RFE(estimator=feature_selection_model, n_features_to_select=n_features_to_select)),
                             ('training_model', MultiOutputRegressor(training_model))])
        super().__init__(name, pipeline, features_data, target_data)

class FRegressionFeatureSlectionPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 50):
        pipeline = Pipeline([('training_model', MultiOutputRegressor(Pipeline([('feature_selection_k_best', SelectKBest(f_regression, k)),
                                                                               ('training_model', training_model)])))])
        super().__init__(name, pipeline, features_data, target_data)


class MutualInfoRegressionFeatureSlectionPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 50):
        pipeline = Pipeline([('training_model', MultiOutputRegressor(Pipeline([('feature_selection_k_best', SelectKBest(mutual_info_regression, k)),
                                                                               ('training_model', training_model)])))])
        super().__init__(name, pipeline, features_data, target_data)


class PartialPCAPipelineRunner(TrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, n_components: float = 6, extra_x=None):
        pipeline = Pipeline([('pca', PcaPartial(n_components=n_components, extra_x=extra_x)),
                             ('training_model', training_model)])
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
            self.unsupervised_data = unsupervised_data
            self.model = model

        def _add_unsupervised_data(self, X, y):
            self.model.fit(X, y)
            y_unsupervised_data = self.model.predict(self.unsupervised_data)

            return X.append(self.unsupervised_data), y.append(pd.DataFrame(y_unsupervised_data, index=self.unsupervised_data.index, columns=y.columns))

        def fit(self, X, y):
            X_with_unsupervised_data, y_with_unsupervised_data = self._add_unsupervised_data(X, y)
            self.model.fit(X_with_unsupervised_data, y_with_unsupervised_data)

            return self

        def predict(self, X):
            return self.model.predict(X)

    def __init__(self, name: str, model, supervised_features_data: pd.DataFrame, target_data: pd.DataFrame, un_supervised_features_data: pd.DataFrame):
        pipeline = Pipeline([('model', self.SemisupervisedModelWrapper(model, un_supervised_features_data))])
        super().__init__(name, pipeline, supervised_features_data, target_data)


class ClassificationTrainingRunner(object):
    RANDOM_STATE = 10
    VERBOSITY = 0

    def __init__(self, name: str, pipeline: Pipeline, features_data: pd.DataFrame, target_data: pd.DataFrame):
        self._name = name
        self.pipeline = pipeline
        self.X = features_data
        self.y = target_data

    def get_classification_matrix(self, target):
        predicted_classification_data = {}

        for column in self.y.columns:
            current_y = self.y[column]

            self.pipeline.fit(self.X, current_y)
            predicted_classification_data[column] = self.pipeline.predict(target)

        predicted_mutation_matrix = pd.DataFrame.from_dict(predicted_classification_data)
        predicted_mutation_matrix.index = target.index

        return predicted_mutation_matrix

    def __str__(self):
        return self._name


class MannWhtUCorrelationMutationPipelineRunner(ClassificationTrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 10):
        mannwhtu_corr_function = mutation_matrix_utils.corr_function_generator(mannwhitneyu)

        pipeline = Pipeline([('feature_selection', SelectKBest(mannwhtu_corr_function, k=k),),('model', training_model)])
        super().__init__(name, pipeline, features_data, target_data)


class FOneWayCorrelationMutationPipelineRunner(ClassificationTrainingRunner):
    def __init__(self, name: str, training_model, features_data: pd.DataFrame, target_data: pd.DataFrame, k: int = 10):
        f_oneway_corr_function = mutation_matrix_utils.corr_function_generator(f_oneway)

        pipeline = Pipeline([('feature_selection', SelectKBest(f_oneway_corr_function, k=k),),('model', training_model)])
        super().__init__(name, pipeline, features_data, target_data)