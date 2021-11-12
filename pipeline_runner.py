
from utils.data_parser import ResourcesPath, DataTransformation
from utils.pipeline_utils.training_runner import SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)

    trainign_model = GradientBoostingRegressor()
    spearman_runner = SpearmanCorrelationPipelineRunner(trainign_model, beat_rnaseq, beat_drug)

    feature_selection_model = DecisionTreeRegressor()
    dicision_tree_feature_selection = ModelFeatureSlectionPipelineRunner(trainign_model, feature_selection_model, beat_rnaseq, beat_drug)

    print("Running on spearman")
    # spearman_results = spearman_runner.run_cross_validation(number_of_splits=5, number_of_repeats=1)

    print("Running on dicision tree")
    dicision_tree_feature_selection_results = dicision_tree_feature_selection.run_cross_validation(number_of_splits=5, number_of_repeats=1)

    # print(spearman_results)
    print(dicision_tree_feature_selection_results)

    print("Dumping spearman results")
    # spearman_results.to_csv("spearman_results.csv")

    print("Dumping dicision tree results")
    dicision_tree_feature_selection_results.to_csv("dicision_tree_feature_selection_results.csv")



if __name__ == '__main__':
    main()