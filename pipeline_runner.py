import pandas as pd
import numpy as np
import seaborn as sns

from utils.data_parser import ResourcesPath, DataTransformation, SubmissionFolds
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner, PartialPCAPipelineRunner, SemisupervisedPipelineRunner,
    FRegressionFeatureSlectionPipelineRunner, MutualInfoRegressionFeatureSlectionPipelineRunner, RFEFeatureSlectionPipelineRunner,
    FOneWayCorrelationMutationPipelineRunner, MannWhtUCorrelationMutationPipelineRunner)
from utils.results_logger import ResultsLogger
from utils.mutation_matrix_utils import calculate_mutation_drug_correlation_matrix

from sklearn.linear_model import MultiTaskLasso, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import mean_squared_error, r2_score


def run_cv(runner):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=5)
    print(f"{runner} results:")
    print(f"- CV training results: \n\t{results['train_score']}, mean={np.mean(results['train_score'])}")
    print(f"- CV test results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")


def run_cv_and_save_estimated_results(runner, cv, results_logger):
    print(f">>> Running on \'{runner}\':")
    results = runner.run_cross_validation(cv=cv, return_estimator=True)
    print(f"- CV training results: \n\t{results['train_score']}, mean={np.mean(results['train_score'])}")
    print(f"- CV test results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")
    estimated_results = pd.DataFrame()
    mse_folds = []
    r2_folds = []
    for i, (_, test_indexes) in enumerate(cv):
        test_patients = runner.X.iloc[test_indexes]
        results_true = runner.y.iloc[test_indexes]
        results_pred = pd.DataFrame(results['estimator'][i].predict(test_patients),
            index=test_patients.index, columns=results_true.columns)
        estimated_results = estimated_results.append(results_pred)
        mse_folds.append(mean_squared_error(results_true, results_pred, multioutput='raw_values'))
        r2_folds.append(r2_score(results_true, results_pred, multioutput='raw_values'))

    output_file_name = results_logger.get_path_in_dir(f'{runner}_results.tsv')
    print(f"- Writing estimated results to '{output_file_name}'... ", end='')
    estimated_results.T.to_csv(output_file_name, sep='\t')
    print('Done.')
    results_logger.add_result_to_csv([output_file_name, *results['train_score'], np.mean(results['train_score']),
        *results['test_score'], np.mean(results['test_score'])])

    mse_amtrix_file_name = results_logger.get_path_in_dir(f'{runner}_mse.csv')
    print(f"- Writing mse results to '{mse_amtrix_file_name}'... ", end='')
    mse_matrix = np.row_stack(mse_folds)
    mse_dataframe = pd.DataFrame(mse_matrix, index=pd.Index(list(range(len(cv))), name='fold'),
        columns=runner.y.columns)
    mse_dataframe.loc['mean'] = mse_dataframe.mean()
    mse_dataframe.T.to_csv(mse_amtrix_file_name)
    print('Done.')
    r2_matrix_file_name = results_logger.get_path_in_dir(f'{runner}_r2.csv')
    print(f"- Writing r^2 results to '{r2_matrix_file_name}'... ", end='')
    r2_matrix = np.row_stack(r2_folds)
    r2_dataframe = pd.DataFrame(r2_matrix, index=pd.Index(list(range(len(cv))), name='fold'),
        columns=runner.y.columns)
    r2_dataframe.loc['mean'] = r2_dataframe.mean()
    r2_dataframe.T.to_csv(r2_matrix_file_name)
    print('Done.')
    return mse_dataframe.T, r2_dataframe.T


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
        # FRegressionFeatureSlectionPipelineRunner('FRegressionFeatureSlectionPipelineRunner', GradientBoostingRegressor(), beat_rnaseq, beat_drug),
        # MutualInfoRegressionFeatureSlectionPipelineRunner('MutualInfoRegressionFeatureSlectionPipelineRunner', GradientBoostingRegressor(), beat_rnaseq, beat_drug),
        # FRegressionFeatureSlectionPipelineRunner('FRegressionFeatureSlectionPipelineRunnerHuber', HuberRegressor(mModelFeatureSlectionPipelineRunnerax_iter=10000, alpha=0.3), beat_rnaseq, beat_drug)
        # RFEFeatureSlectionPipelineRunner('DecisionTree GradientBoostingRegressor', GradientBoostingRegressor(), DecisionTreeRegressor(), beat_rnaseq, beat_drug)
    ]

    with ResultsLogger('task1') as results_logger:
        for model in task1_models:
            mse, r2 = run_cv_and_save_estimated_results(model, subbmission2_folds, results_logger)
            sns.displot(mse.mean(axis=1)).set_xlabels('MSE values').savefig(f'results/{model}_mse_dist.png')
            sns.displot(r2.mean(axis=1)).set_xlabels('r2 values').savefig(f'results/{model}_r2_dist.png')


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
        SemisupervisedPipelineRunner('SemisupervisedPipelineRunner', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug, tcga_rnaseq)
    ]

    with ResultsLogger('task2') as results_logger:
        for model in task2_models:
            run_cv_and_save_estimated_results(model, subbmission2_folds, results_logger)

def task3(beat_rnaseq, tcga_rnaseq, beat_drug, tcga_mutations):

    def _get_mutation_beat_patients():
        return [parient.replace('\n', '') for parient in open(ResourcesPath.DRUG_MUT_COR_LABELS.get_path(), 'r').readlines()]

    def _output_results(drug_mutation_corr_matrix, beat_drug, results_logger, model_name):
        drug_mutation_corr_matrix.index = beat_drug.columns
        drug_mutation_corr_matrix = drug_mutation_corr_matrix.fillna(0)

        drug_mutation_corr_matrix.to_csv(results_logger.get_path_in_dir(f"{model_name}.csv"))
        real_mut_drug_corr_matrix = ResourcesPath.DRUG_MUT_COR.get_dataframe(False)

        results_logger.add_result_to_csv([model_name, mean_squared_error(drug_mutation_corr_matrix.loc[real_mut_drug_corr_matrix.index], real_mut_drug_corr_matrix)])

    intersecting_gene_names = beat_rnaseq.columns.intersection(tcga_rnaseq.columns)
    mutation_beat_indeies = _get_mutation_beat_patients()

    beat_rnaseq = beat_rnaseq.loc[mutation_beat_indeies, intersecting_gene_names]
    tcga_rnaseq = tcga_rnaseq.loc[:, intersecting_gene_names]

    task3_models = [
        FOneWayCorrelationMutationPipelineRunner("F One Way Correlation Mutation", GradientBoostingClassifier(n_estimators=10), tcga_rnaseq, tcga_mutations, 10),
        FOneWayCorrelationMutationPipelineRunner("F One Way Correlation Mutation high tol", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations, 10),
        MannWhtUCorrelationMutationPipelineRunner("Mann Wht U Correlation Mutation", GradientBoostingClassifier(n_estimators=10), tcga_rnaseq, tcga_mutations, 10),
        MannWhtUCorrelationMutationPipelineRunner("Mann Wht U Correlation Mutation high tol", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations, 10)
    ]

    with ResultsLogger('task3', ["model", "mse"]) as results_logger:
        for model in task3_models:
            beat_mutations_predictions = model.get_classification_matrix(beat_rnaseq)
            drug_mutation_corr_matrix = calculate_mutation_drug_correlation_matrix(beat_mutations_predictions, beat_drug)
            _output_results(drug_mutation_corr_matrix, beat_drug, results_logger, str(model))

def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    tcga_rnaseq = ResourcesPath.TCGA_RNA.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)
    beat_drug_without_missing_IC50 = ResourcesPath.BEAT_DRUG.get_dataframe(False, DataTransformation.log10)
    tcga_mutations = ResourcesPath.TCGA_MUT.get_dataframe()

    subbmission2_folds = SubmissionFolds.get_submission2_beat_folds()

    task1(beat_rnaseq.copy(), beat_drug.copy(), subbmission2_folds)
    task2(beat_rnaseq.copy(), tcga_rnaseq.copy(), beat_drug.copy(), subbmission2_folds)
    task3(beat_rnaseq.copy(), tcga_rnaseq.copy(), beat_drug_without_missing_IC50.copy(), tcga_mutations.copy())
    # for drug_name in beat_drug.columns:
        # sns.displot(beat_drug[drug_name], kind="kde").savefig(f'results/{drug_name}_dist.png')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()
