import re
import os

import pandas as pd
import numpy as np

from utils.data_parser import ResourcesPath, DataTransformation, SubmissionFolds
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner)
from config import path_consts

from sklearn.linear_model import MultiTaskLasso, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


# Based on: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
def camel_case_to_snake_case(s):
  s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
  s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
  return re.sub('(\s+)', '_', s)


def run_cv(runner):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=5)
    print(f"{runner} results:")
    print(f"{results['test_score']}, mean={np.mean(results['test_score'])}")


def run_cv_and_save_estimated_results(runner, cv):
    print(f">>> Running on \'{runner}\':")
    results = runner.run_cross_validation(cv=cv, return_estimator=True)
    print(f"- CV results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")
    estimated_results = pd.DataFrame()
    for i, (_, test_indexes) in enumerate(cv):
        test_patients = runner.X.iloc[test_indexes]
        test_results = results['estimator'][i].predict(test_patients)
        estimated_results = estimated_results.append(pd.DataFrame(test_results, index=test_patients.index))
    estimated_results.columns = runner.y.columns

    output_file_name = camel_case_to_snake_case(f'{runner}_results.tsv')
    print(f"- Writing estimated results to '{output_file_name}'... ", end='')
    estimated_results.T.to_csv(os.path.join(path_consts.RESULTS_FOLDER_PATH, output_file_name), sep='\t')
    print('Done.')


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)
    subbmission2_folds = SubmissionFolds.get_submission2_beat_folds()
    run_subbmission2 = lambda runner: run_cv_and_save_estimated_results(runner, subbmission2_folds)

    # [-0.38661757 -0.40991111 -0.45967798 -0.49646509 -0.45710408], mean=-0.44195516728436035
    huber_pca_runner1 = PCAPipelineRunner('PCAHuberRegressor',
        MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=30)
    run_subbmission2(huber_pca_runner1)

    # [-0.41508957 -0.43362979 -0.45954428 -0.5094258  -0.50581295], mean=-0.4647004792869687
    huber_pca_runner2 = PCAPipelineRunner('PCAHuberRegressor2',
        MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=6)
    run_subbmission2(huber_pca_runner2)

    # [-0.39579522 -0.4154555  -0.4578679  -0.50969477 -0.46243136], mean=-0.4482489479071742
    huber_pca_runner3 = PCAPipelineRunner('PCAHuberRegressor3',
        MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=50)
    run_subbmission2(huber_pca_runner3)

    # [-0.45381191 -0.47102307 -0.49968704 -0.52427128 -0.53961544], mean=-0.49768174829512296
    huber_pca_runner4 = PCAPipelineRunner('PCAHuberRegressor4',
        MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=100)
    run_subbmission2(huber_pca_runner4)

    # huber_raw_runner = RawPipelineRunner('RawHuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug)
    # run_subbmission2(huber_raw_runner)

    # [-0.40169296 -0.41632093 -0.43050571 -0.46996061 -0.47875695], mean=-0.4394474314129019
    pca_lasso_runner = PCAPipelineRunner('PCAMultiTaskLasso',
        MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    run_subbmission2(pca_lasso_runner)

    # [-0.36321811 -0.39005344 -0.40093766 -0.44930264 -0.43693246], mean=-0.40808886428070645
    lasso_runner = RawPipelineRunner('Raw MultiTaskLasso',
        MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    run_subbmission2(lasso_runner)

    # [-0.36592024 -0.3903679  -0.4041547  -0.45183953 -0.43228437], mean=-0.40891334702597765
    lasso_runner2 = RawPipelineRunner('Raw MultiTaskLasso2',
        MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.7), beat_rnaseq, beat_drug)
    run_subbmission2(lasso_runner2)

    # [-0.3637785  -0.38994446 -0.40186929 -0.45009532 -0.43253616], mean=-0.4076447450068906
    lasso_runner3 = RawPipelineRunner('Raw MultiTaskLasso3',
        MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug)
    run_subbmission2(lasso_runner3)

    # [-0.40258642 -0.41668917 -0.43136205 -0.47045855 -0.48091484], mean=-0.4404022063313297
    ridge = PCAPipelineRunner('PCA MultiOutputRegressor Ridge',
        MultiOutputRegressor(Ridge(random_state=10, max_iter=10000, alpha=1)), beat_rnaseq, beat_drug)
    run_subbmission2(ridge)

    # training_model = GradientBoostingRegressor()
    # spearman_runner = SpearmanCorrelationPipelineRunner(training_model, beat_rnaseq, beat_drug)
    # spearman_runner_results = spearman_runner.run_cross_validation(cv=5)
    # print("SpearmanCorr -> GradientBoostingRegressor results ")
    # print(f"{spearman_runner_results['test_score']}, mean={np.mean(spearman_runner_results['test_score'])}")

    # [-0.48706888 -0.51880403 -0.51332265 -0.53211253 -0.55660673], mean=-0.5215829660994237
    training_model = GradientBoostingRegressor()
    feature_selection_model = DecisionTreeRegressor()
    dicision_tree_feature_selection = ModelFeatureSlectionPipelineRunner('DecisionTree GradientBoostingRegressor',
        training_model, feature_selection_model, beat_rnaseq, beat_drug)
    run_subbmission2(dicision_tree_feature_selection)

    # [-0.40258673 -0.41668887 -0.43136464 -0.47045988 -0.48091307], mean=-0.4404026388490522
    linear_regression_runner = PCAPipelineRunner('PCA LinearRegression',
        MultiOutputRegressor(LinearRegression()), beat_rnaseq, beat_drug)
    run_subbmission2(linear_regression_runner)

    # [-0.38023087 -0.39994    -0.41673743 -0.46159207 -0.44569175], mean=-0.42083842590546927
    regressor_chain_runner = PCAPipelineRunner('PCA RegressorChain',
        RegressorChain(Lasso(alpha=0.7), order='random', random_state=42), beat_rnaseq, beat_drug, n_components=40)
    run_subbmission2(regressor_chain_runner)

    # [-0.41492778 -0.44420791 -0.45843145 -0.50111789 -0.48496732], mean=-0.4607304682176365
    regressor_chain_runner2 = RawPipelineRunner('Raw RegressorChain',
        RegressorChain(Lasso(alpha=1.0), order='random', random_state=10), beat_rnaseq, beat_drug)
    run_subbmission2(regressor_chain_runner2)

    # [-0.4274169  -0.45143854 -0.45497157 -0.52702544 -0.48108416], mean=-0.46838731956860447
    pca_gradient_boosting_regressor = PCAPipelineRunner('PCA GradientBoostingRegressor',
        MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50)
    run_subbmission2(pca_gradient_boosting_regressor)

    # [-0.40166475 -0.416212   -0.44313964 -0.47470732 -0.4990718 ], mean=-0.4469591012873904
    gradient_boosting_regressor = RawPipelineRunner('Raw -> GradientBoostingRegressor',
        MultiOutputRegressor(GradientBoostingRegressor(random_state=42, max_features='log2')), beat_rnaseq, beat_drug)
    run_subbmission2(gradient_boosting_regressor)

    # [-0.40031892 -0.42277723 -0.4361124  -0.509762   -0.47089545], mean=-0.4479731964725596
    random_forest_regressor = PCAPipelineRunner('PCA -> RandomForestRegressor',
        MultiOutputRegressor(RandomForestRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50)
    run_subbmission2(random_forest_regressor)

if __name__ == '__main__':
    main()
