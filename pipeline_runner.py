import numpy as np
import seaborn as sns
import pandas as pd
import argparse

from utils.data_parser import ResourcesPath, DataTransformation, SubmissionFolds
from utils.classifier_results_utils import analyze_classifier_roc, analyze_classifier_pr
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner, PartialPCAPipelineRunner, SemisupervisedPipelineRunner,
    FRegressionFeatureSlectionPipelineRunner, MutualInfoRegressionFeatureSlectionPipelineRunner, RFEFeatureSlectionPipelineRunner,
    FOneWayCorrelationMutationPipelineRunner, MannWhtUCorrelationMutationPipelineRunner, SpearmanCorrelationClustingPipelineRunner,
    BaysianFeatureSelectionMutationPipelineRunner, RawClassificationTrainingRunner, Chi2Selector)
from utils.results_logger import ResultsLogger
from utils.mutation_matrix_utils import calculate_mutation_drug_correlation_matrix

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import MultiTaskLasso, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.feature_selection import VarianceThreshold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import mean_squared_error


class Tasks(Enum):
    task1 = '1'
    task2 = '2'
    task3 = '3'


def _parse_args():
    parser = argparse.ArgumentParser(description='Training pipeline runner')
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('task', type=Tasks, help='The task to be trained')
    train_parser.set_defaults(func=train_models)
    dump_parser = subparsers.add_parser('dump')
    dump_parser.add_argument('task', type=Tasks, help='Dump the best task')
    dump_parser.set_defaults(func=dump_models)
    return parsed_args


def get_beat_rnaseq():
    return ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)


def get_tcga_rnaseq():
    return ResourcesPath.TCGA_RNA.get_dataframe(True, DataTransformation.log2)


def get_beat_drug():
    return ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)


def get_beat_drug_without_missing_IC50():
    return ResourcesPath.BEAT_DRUG.get_dataframe(False, DataTransformation.log10)


def get_tcga_mutations():
    return ResourcesPath.TCGA_MUT.get_dataframe()


def get_subbmission2_folds():
    return SubmissionFolds.get_submission2_beat_folds()


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
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso Precalculated Clustering', [MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster and Random Forest Precalculated Clustering', [RandomForestRegressor(random_state=10, max_depth=5) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso (feature selection) and Random Forest Pre calculated caluster', [Pipeline([('feature_selection_k_best', SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.2), max_features=1000)),
                             ('training_model', RandomForestRegressor(random_state=10, max_depth=5))]) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso (feature selection) and Random Forest 2', [Pipeline([('feature_selection_k_best', SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), max_features=1000)),
                             ('training_model', RandomForestRegressor(random_state=10, max_depth=5, n_estimators=300))]) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        # -0.40017291529307164
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso (feature selection) and Random Forest 3', [Pipeline([('feature_selection_k_best', SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), max_features=200)),
                             ('training_model', RandomForestRegressor(random_state=10, max_depth=5, n_estimators=500))]) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        # -0.399366508767371
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso (feature selection) and Random Forest 4', [Pipeline([('feature_selection_k_best', SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), max_features=100)),
                             ('training_model', RandomForestRegressor(random_state=10, max_depth=7, n_estimators=500))]) for i in range(3)], beat_rnaseq, beat_drug, 3, True),
        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso (feature selection) and Random Forest 7 cluster',
         [Pipeline([("var_threshold", VarianceThreshold(0.5)),
         ('feature_selection_k_best', SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.3), max_features=5000)),
         ('training_model', RandomForestRegressor(random_state=10, max_depth=12, n_estimators=200))]) for i in range(7)], beat_rnaseq, beat_drug, 7),

        SpearmanCorrelationClustingPipelineRunner('Drug Cluster Multi Lasso and Chi2 (feature selection) and Random Forest 3 cluster',
         [Pipeline([("var_threshold", VarianceThreshold(0.5)),
         ('feature_selection_k_best', FeatureUnion([("select_from_multitasklasso", SelectFromModel(estimator=MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.3), max_features=5000)),
                                                    ("Chi2", Chi2Selector())])),
         ('training_model', RandomForestRegressor(random_state=10, max_depth=10, n_estimators=200))]) for i in range(3)], beat_rnaseq, beat_drug, 3),
        RawPipelineRunner('Raw MultioutLasso', MultiOutputRegressor(Lasso(random_state=10, max_iter=10000, alpha=1.0)), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultioutLasso2', MultiOutputRegressor(Lasso(random_state=10, max_iter=10000, alpha=0.7)), beat_rnaseq, beat_drug),
        RawPipelineRunner('Raw MultioutLasso3', MultiOutputRegressor(Lasso(random_state=10, max_iter=10000, alpha=0.8)), beat_rnaseq, beat_drug),
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
        # Drugs are cols and mutations are rows
        drug_mutation_corr_matrix.T.to_csv(results_logger.get_path_in_dir(f"{model_name}.tsv"), sep='\t')
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
        RawClassificationTrainingRunner("logistic regression", LogisticRegression(), tcga_rnaseq, tcga_mutations),
        RawClassificationTrainingRunner("logistic regression c0.7", LogisticRegression(C=0.7), tcga_rnaseq, tcga_mutations), # best 0.007489285
        RawClassificationTrainingRunner("logistic regression c0.7 10000 iter", LogisticRegression(C=0.7, max_iter=10000), tcga_rnaseq, tcga_mutations),
        RawClassificationTrainingRunner("logistic regression c0.8", LogisticRegression(C=0.8), tcga_rnaseq, tcga_mutations),
        RawClassificationTrainingRunner("logistic regression c0.5", LogisticRegression(C=0.5), tcga_rnaseq, tcga_mutations),
        RawClassificationTrainingRunner("logistic regression c0.2", LogisticRegression(C=0.2), tcga_rnaseq, tcga_mutations),
        RawClassificationTrainingRunner("logistic regression 10000 iter", LogisticRegression(max_iter=10000), tcga_rnaseq, tcga_mutations),
        RawClassificationTrainingRunner("logistic regression l1", LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5), tcga_rnaseq, tcga_mutations),
        FOneWayCorrelationMutationPipelineRunner("F One Way Correlation Mutation", GradientBoostingClassifier(n_estimators=10), tcga_rnaseq, tcga_mutations, 10),
        FOneWayCorrelationMutationPipelineRunner("F One Way Correlation Mutation high tol", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations, 10),
        MannWhtUCorrelationMutationPipelineRunner("Mann Wht U Correlation Mutation", GradientBoostingClassifier(n_estimators=10), tcga_rnaseq, tcga_mutations, 10),
        MannWhtUCorrelationMutationPipelineRunner("Mann Wht U Correlation Mutation high tol", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations, 10),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=10, tol=0.01), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and KNeighborsClassifier 3 neighbors", KNeighborsClassifier(3), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and KNeighborsClassifier 5 neighbors", KNeighborsClassifier(5), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and KNeighborsClassifier 10 neighbors ", KNeighborsClassifier(10), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and KNeighborsClassifier distance", KNeighborsClassifier(5, weights="distance"), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and DecisionTreeClassifier", DecisionTreeClassifier(max_depth=20), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and RandomForestClassifier 3 depth", RandomForestClassifier(max_depth=3, n_estimators=10), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and RandomForestClassifier 5 depth", RandomForestClassifier(max_depth=5, n_estimators=10), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and RandomForestClassifier 10 depth", RandomForestClassifier(max_depth=10, n_estimators=10), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and LogisticRegression l2", LogisticRegression(), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and LogisticRegression l2 liblinear", LogisticRegression(penalty="l2", solver="newton-cg"), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and LogisticRegression l1 liblinear", LogisticRegression(penalty="l1", solver="liblinear"), tcga_rnaseq, tcga_mutations),
        BaysianFeatureSelectionMutationPipelineRunner("Reactome feature selection and LogisticRegression l1 saga", LogisticRegression(penalty="l1", solver="saga"), tcga_rnaseq, tcga_mutations),
    ]

    with ResultsLogger('task3', ["model", "mse"]) as results_logger:
        for model in task3_models:
            beat_mutations_predictions = model.get_classification_matrix(beat_rnaseq_only_mutaions)
            drug_mutation_corr_matrix = calculate_mutation_drug_correlation_matrix(beat_mutations_predictions, beat_drug_without_missing_IC50)
            _output_results(drug_mutation_corr_matrix, results_logger, str(model))
            analyze_classifier_roc(model, results_logger.get_path_in_dir(f"{str(model)}_MUTATION_NAME_roc_analysis.png"))
            analyze_classifier_pr(model, results_logger.get_path_in_dir(f"{str(model)}_MUTATION_NAME_pr_analysis.png"))

        task1_selected_model = RawPipelineRunner('Raw MultiTaskLasso3', MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug)
        _output_task1_to_task3_results(task1_selected_model, tcga_rnaseq, results_logger)
    print('<<<<<<<< TASK3 END >>>>>>>>')


def train_models(args):
    if args.task == Tasks.task1:
        task1(get_beat_rnaseq(), get_beat_drug(), get_subbmission2_folds())
    elif args.task == Tasks.task2:
        task2(get_beat_rnaseq(), get_tcga_rnaseq(), get_beat_drug(), get_subbmission2_folds())
    elif args.task == Tasks.task3:
        task3(get_beat_rnaseq(), get_tcga_rnaseq(), get_beat_drug(), get_beat_drug_without_missing_IC50(), get_tcga_mutations())
    else:
        raise RuntimeError("Invalid task")


def dump_models(args):
    if args.task == Tasks.task1:
        model = ModelFeatureSlectionPipelineRunner(
            'Multi Lasso (feature selection) and Random Forest 4',
            RandomForestRegressor(random_state=10, max_depth=7, n_estimators=500),
            MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8),
            get_beat_rnaseq(),
            get_beat_drug(),
            model_is_multitask=True
        )
        model.dump_train_model("task1", True)
    elif args.task == Tasks.task2:
        model = ModelFeatureSlectionPipelineRunner(
            'Multi Lasso (feature selection) and Random Forest 4',
            RandomForestRegressor(random_state=10, max_depth=7, n_estimators=500),
            MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8),
            get_beat_rnaseq(),
            get_beat_drug(),
            model_is_multitask=True
        )
        model.dump_train_model("task2", True)
    elif args.task == Tasks.task3:
        model = RawClassificationTrainingRunner("logistic regression c0.8", LogisticRegression(C=0.8), tcga_rnaseq, tcga_mutations),
        model.dump_train_model("task3", True)
    else:
        raise RuntimeError("Invalid task")


def main():
    parsed_args = _parse_args()
    args.func(args)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
