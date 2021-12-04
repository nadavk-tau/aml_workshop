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

    models = [
        # [-0.38661757 -0.40991111 -0.45967798 -0.49646509 -0.45710408], mean=-0.44195516728436035
        PCAPipelineRunner('PCAHuberRegressor', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=30),
        # [-0.41508957 -0.43362979 -0.45954428 -0.5094258  -0.50581295], mean=-0.4647004792869687
        PCAPipelineRunner('PCAHuberRegressor2', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=6),
        # [-0.39579522 -0.4154555  -0.4578679  -0.50969477 -0.46243136], mean=-0.4482489479071742
        PCAPipelineRunner('PCAHuberRegressor3', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=50),
        # [-0.45381191 -0.47102307 -0.49968704 -0.52427128 -0.53961544], mean=-0.49768174829512296
        PCAPipelineRunner('PCAHuberRegressor4', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=100),
        # RawPipelineRunner('RawHuberRegressor', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug),
        # [-0.40169296 -0.41632093 -0.43050571 -0.46996061 -0.47875695], mean=-0.4394474314129019
        PCAPipelineRunner('PCAMultiTaskLasso', MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug),
        # [-0.36321811 -0.39005344 -0.40093766 -0.44930264 -0.43693246], mean=-0.40808886428070645
        RawPipelineRunner('Raw MultiTaskLasso', MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug),
        # [-0.36592024 -0.3903679  -0.4041547  -0.45183953 -0.43228437], mean=-0.40891334702597765
        RawPipelineRunner('Raw MultiTaskLasso2', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.7), beat_rnaseq, beat_drug),
        # [-0.3637785  -0.38994446 -0.40186929 -0.45009532 -0.43253616], mean=-0.4076447450068906
        RawPipelineRunner('Raw MultiTaskLasso3', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug),
        # [-0.40258642 -0.41668917 -0.43136205 -0.47045855 -0.48091484], mean=-0.4404022063313297
        PCAPipelineRunner('PCA MultiOutputRegressor Ridge', MultiOutputRegressor(Ridge(random_state=10, max_iter=10000, alpha=1)), beat_rnaseq, beat_drug),
        # SpearmanCorrelationPipelineRunner(GradientBoostingRegressor(), beat_rnaseq, beat_drug),
        # [-0.48706888 -0.51880403 -0.51332265 -0.53211253 -0.55660673], mean=-0.5215829660994237
        ModelFeatureSlectionPipelineRunner('DecisionTree GradientBoostingRegressor', GradientBoostingRegressor(), DecisionTreeRegressor(), beat_rnaseq, beat_drug),
        # [-0.40258673 -0.41668887 -0.43136464 -0.47045988 -0.48091307], mean=-0.4404026388490522
        PCAPipelineRunner('PCA LinearRegression', MultiOutputRegressor(LinearRegression()), beat_rnaseq, beat_drug),
        # [-0.38023087 -0.39994    -0.41673743 -0.46159207 -0.44569175], mean=-0.42083842590546927
        PCAPipelineRunner('PCA RegressorChain', RegressorChain(Lasso(alpha=0.7), order='random', random_state=42), beat_rnaseq, beat_drug, n_components=40),
        RawPipelineRunner('Raw RegressorChain', RegressorChain(Lasso(alpha=1.0), order='random', random_state=10), beat_rnaseq, beat_drug),
        # [-0.4274169  -0.45143854 -0.45497157 -0.52702544 -0.48108416], mean=-0.46838731956860447
        PCAPipelineRunner('PCA GradientBoostingRegressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50),
        # [-0.40166475 -0.416212   -0.44313964 -0.47470732 -0.4990718 ], mean=-0.4469591012873904
        RawPipelineRunner('Raw -> GradientBoostingRegressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42, max_features='log2')), beat_rnaseq, beat_drug),
        # [-0.40031892 -0.42277723 -0.4361124  -0.509762   -0.47089545], mean=-0.4479731964725596
        PCAPipelineRunner('PCA -> RandomForestRegressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50)
    ]

    for model in models:
        run_cv_and_save_estimated_results(model, subbmission2_folds)
    
    
if __name__ == '__main__':
    main()
