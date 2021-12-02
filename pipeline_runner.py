
import numpy as np

from utils.data_parser import ResourcesPath, DataTransformation
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner)

from sklearn.linear_model import MultiTaskLasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def run_cv(runner):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=5)
    print(f"{runner} results:")
    print(f"{results['test_score']}, mean={np.mean(results['test_score'])}")


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)

    lasso_runner = PCAPipelineRunner('PCA -> MultiTaskLasso',
        MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    run_cv(lasso_runner)

    # lasso_runner2 = RawPipelineRunner('Raw MultiTaskLasso',
    #     MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    # run_cv(lasso_runner2)

    training_model = GradientBoostingRegressor()
    # spearman_runner = SpearmanCorrelationPipelineRunner(training_model, beat_rnaseq, beat_drug)
    # spearman_runner_results = spearman_runner.run_cross_validation(cv=5)
    # print("SpearmanCorr -> GradientBoostingRegressor results ")
    # print(f"{spearman_runner_results['test_score']}, mean={np.mean(spearman_runner_results['test_score'])}")

    feature_selection_model = DecisionTreeRegressor()
    dicision_tree_feature_selection = ModelFeatureSlectionPipelineRunner('DecisionTree -> GradientBoostingRegressor',
        training_model, feature_selection_model, beat_rnaseq, beat_drug)
    run_cv(dicision_tree_feature_selection)

    linear_regression_runner = PCAPipelineRunner('PCA -> LinearRegression',
        MultiOutputRegressor(LinearRegression()), beat_rnaseq, beat_drug)
    run_cv(linear_regression_runner)

if __name__ == '__main__':
    main()