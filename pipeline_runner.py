
import numpy as np

from utils.data_parser import ResourcesPath, DataTransformation
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner, PCAPipelineRunner)

from sklearn.linear_model import MultiTaskLasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)

    lasso_runner = PCAPipelineRunner(MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    lasso_runner_results = lasso_runner.run_cross_validation(cv=5)
    print("PCA -> MultiTaskLasso results:")
    print(f"{lasso_runner_results['test_score']}, mean={np.mean(lasso_runner_results['test_score'])}")

    training_model = GradientBoostingRegressor()
    spearman_runner = SpearmanCorrelationPipelineRunner(training_model, beat_rnaseq, beat_drug)

    feature_selection_model = DecisionTreeRegressor()
    dicision_tree_feature_selection = ModelFeatureSlectionPipelineRunner(training_model, feature_selection_model, beat_rnaseq, beat_drug)
    dicision_tree_feature_selection_results = dicision_tree_feature_selection.run_cross_validation(cv=5)

    print("DecisionTree -> GradientBoostingRegressor results ")
    print(f"{dicision_tree_feature_selection_results['test_score']}, mean={np.mean(dicision_tree_feature_selection_results['test_score'])}")

if __name__ == '__main__':
    main()