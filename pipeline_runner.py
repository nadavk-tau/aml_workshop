import numpy as np
import seaborn as sns
import pandas as pd

from utils.data_parser import ResourcesPath, DataTransformation, SubmissionFolds
from utils.classifier_results_utils import analyze_classifier
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner, PartialPCAPipelineRunner, SemisupervisedPipelineRunner,
    FRegressionFeatureSlectionPipelineRunner, MutualInfoRegressionFeatureSlectionPipelineRunner, RFEFeatureSlectionPipelineRunner,
    FOneWayCorrelationMutationPipelineRunner, MannWhtUCorrelationMutationPipelineRunner, SpearmanCorrelationClustingPipelineRunner, SplittedPipelineRunner)
from utils.results_logger import ResultsLogger
from utils.mutation_matrix_utils import calculate_mutation_drug_correlation_matrix

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import MultiTaskLasso, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import mean_squared_error


def run_cv(runner):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=5)
    print(f"{runner} results:")
    print(f"- CV training results: \n\t{results['train_score']}, mean={np.mean(results['train_score'])}")
    print(f"- CV test results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")

def run_cv_and_save_estimated_results(runner, cv, results_logger, output_graphs=False):
    print(f">>> Running on \'{runner}\':")
    results = runner.run_cross_validation_and_get_estimated_results(cv=cv)
    print(f"- CV training results: \n\t{results['train_score']}, mean={np.mean(results['train_score'])}")
    print(f"- CV test results: \n\t{results['test_score']}, mean={np.mean(results['test_score'])}")

    full_output_file_name = results_logger.save_csv(f'{runner}_estimated_results.tsv', 'estimated results',
        results['estimated_results'], sep='\t')
    results_logger.add_result_to_cv_results_csv([full_output_file_name, *results['train_score'], np.mean(results['train_score']),
        *results['test_score'], np.mean(results['test_score'])])

    results_logger.save_csv(f'{runner}_mse.csv', 'MSE results', results['mse_matrix'])
    results_logger.save_csv(f'{runner}_r2.csv', 'R^2 results', results['r2_matrix'])
    if output_graphs:
        mse_fig = sns.displot(results['mse_matrix'].mean(axis=1)).set_xlabels('MSE values')
        results_logger.save_figure(f'{runner}_mse_dist.png', 'MSE histogram', mse_fig)
        r2_fig = sns.displot(results['r2_matrix'].mean(axis=1)).set_xlabels('R^2 values')
        results_logger.save_figure(f'{runner}_r2_dist.png', 'R^2 histogram', r2_fig)


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
        PCAPipelineRunner('PCA RandomForestRegressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso', [MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster and Random Forest', [RandomForestRegressor(random_state=10, max_depth=5) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso (feature selection) and Random Forest', [Pipeline([('feature_selection_k_best', SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), max_features=1000)),
                             ('training_model', RandomForestRegressor(random_state=10, max_depth=5))]) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        RawPipelineRunner('Raw MultioutLasso', MultiOutputRegressor(Lasso(random_state=10, max_iter=10000, alpha=1.0)), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultioutLasso2', MultiOutputRegressor(Lasso(random_state=10, max_iter=10000, alpha=0.7)), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultioutLasso3', MultiOutputRegressor(Lasso(random_state=10, max_iter=10000, alpha=0.8)), beat_rnaseq, beat_drug),
        SplittedPipelineRunner('Random Foresset and MultiLasso', ),
        # FRegressionFeatureSlectionPipelineRunner('FRegressionFeatureSlectionPipelineRunner', GradientBoostingRegressor(), beat_rnaseq, beat_drug),
        # MutualInfoRegressionFeatureSlectionPipelineRunner('MutualInfoRegressionFeatureSlectionPipelineRunner', GradientBoostingRegressor(), beat_rnaseq, beat_drug),
        # FRegressionFeatureSlectionPipelineRunner('FRegressionFeatureSlectionPipelineRunnerHuber', HuberRegressor(mModelFeatureSlectionPipelineRunnerax_iter=10000, alpha=0.3), beat_rnaseq, beat_drug),
        # RFEFeatureSlectionPipelineRunner('DecisionTree GradientBoostingRegressor', GradientBoostingRegressor(), DecisionTreeRegressor(), beat_rnaseq, beat_drug)
    ]

    print('<<<<<<<< TASK1 BEGIN >>>>>>>>')
    with ResultsLogger('task1') as results_logger:
        for model in task1_models:
            run_cv_and_save_estimated_results(model, subbmission2_folds, results_logger, output_graphs=True)
    print('<<<<<<<< TASK1 END >>>>>>>>')


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

    print('<<<<<<<< TASK2 BEGIN >>>>>>>>')
    with ResultsLogger('task2') as results_logger:
        for model in task2_models:
            run_cv_and_save_estimated_results(model, subbmission2_folds, results_logger)
    print('<<<<<<<< TASK2 END >>>>>>>>')


def task3(beat_rnaseq, tcga_rnaseq, beat_drug, beat_drug_without_missing_IC50, tcga_mutations):

    def _get_mutation_beat_patients():
        return [parient.replace('\n', '') for parient in open(ResourcesPath.DRUG_MUT_COR_LABELS.get_path(), 'r').readlines()]

    def _output_results(drug_mutation_corr_matrix, results_logger, model_name):
        drug_mutation_corr_matrix.to_csv(results_logger.get_path_in_dir(f"{model_name}.csv"))
        real_mut_drug_corr_matrix = ResourcesPath.DRUG_MUT_COR.get_dataframe(False)

        results_logger.add_result_to_cv_results_csv([model_name, mean_squared_error(drug_mutation_corr_matrix.loc[real_mut_drug_corr_matrix.index], real_mut_drug_corr_matrix)])

    def _output_task1_to_task3_results(pipeline_runner, tcga_rnaseq, results_logger):
        pipeline_runner.pipeline.fit(pipeline_runner.X, pipeline_runner.y)
        tcga_drug_predictions = pipeline_runner.pipeline.predict(tcga_rnaseq)
        tcga_drug_predictions = pd.DataFrame(tcga_drug_predictions, columns=pipeline_runner.y.columns, index=tcga_rnaseq.index)

        drug_mutation_corr_matrix = calculate_mutation_drug_correlation_matrix(ResourcesPath.TCGA_MUT.get_dataframe(), tcga_drug_predictions)
        _output_results(drug_mutation_corr_matrix, results_logger, str(pipeline_runner))


    print('<<<<<<<< TASK3 BEGIN >>>>>>>>')
    intersecting_gene_names = beat_rnaseq.columns.intersection(tcga_rnaseq.columns)
    mutation_beat_indeies = _get_mutation_beat_patients()

    beat_rnaseq = beat_rnaseq.loc[:, intersecting_gene_names]
    beat_rnaseq_only_mutaions = beat_rnaseq.loc[mutation_beat_indeies, :]
    tcga_rnaseq = tcga_rnaseq.loc[:, intersecting_gene_names]

    task3_models = [
        FOneWayCorrelationMutationPipelineRunner("F One Way Correlation Mutation", GradientBoostingClassifier(n_estimators=10), tcga_rnaseq, tcga_mutations, 10),
        FOneWayCorrelationMutationPipelineRunner("F One Way Correlation Mutation high tol", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations, 10),
        MannWhtUCorrelationMutationPipelineRunner("Mann Wht U Correlation Mutation", GradientBoostingClassifier(n_estimators=10), tcga_rnaseq, tcga_mutations, 10),
        MannWhtUCorrelationMutationPipelineRunner("Mann Wht U Correlation Mutation high tol", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations, 10)
    ]

    with ResultsLogger('task3', ["model", "mse"]) as results_logger:
        for model in task3_models:
            beat_mutations_predictions = model.get_classification_matrix(beat_rnaseq_only_mutaions)
            drug_mutation_corr_matrix = calculate_mutation_drug_correlation_matrix(beat_mutations_predictions, beat_drug_without_missing_IC50)
            _output_results(drug_mutation_corr_matrix, results_logger, str(model))
            analyze_classifier(model, results_logger.get_path_in_dir(f"{str(model)}_MUTATION_NAME_roc_analysis.png"))

        task1_selected_model = RawPipelineRunner('Raw MultiTaskLasso3', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug)
        _output_task1_to_task3_results(task1_selected_model, tcga_rnaseq, results_logger)

    print('<<<<<<<< TASK3 END >>>>>>>>')


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    tcga_rnaseq = ResourcesPath.TCGA_RNA.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)
    beat_drug_without_missing_IC50 = ResourcesPath.BEAT_DRUG.get_dataframe(False, DataTransformation.log10)
    tcga_mutations = ResourcesPath.TCGA_MUT.get_dataframe()

    subbmission2_folds = SubmissionFolds.get_submission2_beat_folds()

    task1(beat_rnaseq.copy(), beat_drug.copy(), subbmission2_folds)
    # task2(beat_rnaseq.copy(), tcga_rnaseq.copy(), beat_drug.copy(), subbmission2_folds)
    # task3(beat_rnaseq.copy(), tcga_rnaseq.copy(), beat_drug.copy(), beat_drug_without_missing_IC50.copy(), tcga_mutations.copy())



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
