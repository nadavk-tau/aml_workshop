import pandas as pd
import numpy as np

from utils.data_parser import ResourcesPath, DataTransformation, SubmissionFolds
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner, PartialPCAPipelineRunner, SemisupervisedPipelineRunner)
from utils.results_logger import ResultsDir

from sklearn.linear_model import MultiTaskLasso, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


def run_cv(runner):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=5)
    print(f"{runner} results:")
    print(f"- CV training results: \n\t{results['train_score']}, mean={np.mean(results['train_score'])}")
    print(f"- CV test results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")


def run_cv_and_save_estimated_results(runner, cv, results_dir):
    print(f">>> Running on \'{runner}\':")
    results = runner.run_cross_validation(cv=cv, return_estimator=True)
    print(f"- CV training results: \n\t{results['train_score']}, mean={np.mean(results['train_score'])}")
    print(f"- CV test results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")
    estimated_results = pd.DataFrame()
    for i, (_, test_indexes) in enumerate(cv):
        test_patients = runner.X.iloc[test_indexes]
        test_results = results['estimator'][i].predict(test_patients)
        estimated_results = estimated_results.append(pd.DataFrame(test_results, index=test_patients.index))
    estimated_results.columns = runner.y.columns

    output_file_name = results_dir.get_path_in_dir(f'{runner}_results.tsv')
    print(f"- Writing estimated results to '{output_file_name}'... ", end='')
    estimated_results.T.to_csv(output_file_name, sep='\t')
    print('Done.')


def task1(beat_rnaseq, beat_drug, subbmission2_folds):
    task1_models = [
        PCAPipelineRunner('PCAHuberRegressor', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=30),
        PCAPipelineRunner('PCAHuberRegressor2', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=6),
        PCAPipelineRunner('PCAHuberRegressor3', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=50),
        PCAPipelineRunner('PCAHuberRegressor4', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=100),
        # RawPipelineRunner('RawHuberRegressor', MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCAMultiTaskLasso', MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultiTaskLasso', MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultiTaskLasso2', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.7), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultiTaskLasso3', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCA MultiOutputRegressor Ridge', MultiOutputRegressor(Ridge(random_state=10, max_iter=10000, alpha=1)), beat_rnaseq, beat_drug),
        # SpearmanCorrelationPipelineRunner(GradientBoostingRegressor(), beat_rnaseq, beat_drug),
        ModelFeatureSlectionPipelineRunner('DecisionTree GradientBoostingRegressor', GradientBoostingRegressor(), DecisionTreeRegressor(), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCA LinearRegression', MultiOutputRegressor(LinearRegression()), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCA RegressorChain', RegressorChain(Lasso(alpha=0.7), order='random', random_state=42), beat_rnaseq, beat_drug, n_components=40),
        RawPipelineRunner('Raw RegressorChain', RegressorChain(Lasso(alpha=1.0), order='random', random_state=10), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCA GradientBoostingRegressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50),
        RawPipelineRunner('Raw GradientBoostingRegressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42, max_features='log2')), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCA RandomForestRegressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50)
    ]

    with ResultsDir('task1') as results_dir:
        for model in task1_models:
            run_cv_and_save_estimated_results(model, subbmission2_folds, results_dir)


def task2(beat_rnaseq, tcga_rnaseq, beat_drug, subbmission2_folds):
    intersecting_gene_names = beat_rnaseq.columns.intersection(tcga_rnaseq.columns)
    beat_rnaseq = beat_rnaseq.loc[:, intersecting_gene_names]
    tcga_rnaseq = tcga_rnaseq.loc[:, intersecting_gene_names]
    task2_models = [
        PartialPCAPipelineRunner("PartialPCAPipelineRunner lasso", MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug, extra_x=tcga_rnaseq),
        PCAPipelineRunner('PCAMultiTaskLasso', MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultiTaskLasso', MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultiTaskLasso3', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug),
        PCAPipelineRunner('PCA GradientBoostingRegressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50),
        PCAPipelineRunner('PCA RandomForestRegressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50),
        # SemisupervisedPipelineRunner('SemisupervisedPipelineRunner', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug, tcga_rnaseq)
    ]

    with ResultsDir('task2') as results_dir:
        for model in task2_models:
            run_cv_and_save_estimated_results(model, subbmission2_folds, results_dir)


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    tcga_rnaseq = ResourcesPath.TCGA_RNA.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)
    subbmission2_folds = SubmissionFolds.get_submission2_beat_folds()
    task1(beat_rnaseq.copy(), beat_drug.copy(), subbmission2_folds)
    task2(beat_rnaseq.copy(), tcga_rnaseq.copy(), beat_drug.copy(), subbmission2_folds)
    
    
if __name__ == '__main__':
    main()
