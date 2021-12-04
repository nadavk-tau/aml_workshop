
import numpy as np

from joblib import parallel_backend
from utils.data_parser import ResourcesPath, DataTransformation, SubmissionFolds
from utils.pipeline_utils.training_runner import (SpearmanCorrelationPipelineRunner, ModelFeatureSlectionPipelineRunner,
    PCAPipelineRunner, RawPipelineRunner)

from sklearn.linear_model import MultiTaskLasso, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def run_cv(runner):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=5)
    print(f"{runner} results:")
    print(f"{results['test_score']}, mean={np.mean(results['test_score'])}")


def run_subbmission2(runner, cv):
    print(f">>> Running on \'{runner}\'")
    results = runner.run_cross_validation(cv=cv, return_estimator=True)
    print(f"{runner} results:")
    print(f"{results['test_score']}, mean={np.mean(results['test_score'])}")


def main():
    beat_rnaseq = ResourcesPath.BEAT_RNASEQ.get_dataframe(True, DataTransformation.log2)
    beat_drug = ResourcesPath.BEAT_DRUG.get_dataframe(True, DataTransformation.log10)
    subbmission2_folds = SubmissionFolds.get_submission2_beat_folds()

    # # [-0.40960095 -0.45455335 -0.44915175 -0.64133478 -0.35218218], mean=-0.46136460243742483
    # huber_pca_runner1 = PCAPipelineRunner('Raw -> HuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=30)
    # run_cv(huber_pca_runner1)

    # # [-0.41739004 -0.45177341 -0.47154065 -0.72170964 -0.36103101], mean=-0.4846889496414478
    # huber_pca_runner2 = PCAPipelineRunner('Raw -> HuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=6)
    # run_cv(huber_pca_runner2)

    # # [-0.42357851 -0.47222588 -0.45895531 -0.62801165 -0.35544116], mean=-0.46764250349556014
    # huber_pca_runner3 = PCAPipelineRunner('Raw -> HuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=50)
    # run_cv(huber_pca_runner3)

    # [-0.48453742 -0.52254805 -0.51430271 -0.65282478 -0.41325604], mean=-0.5174937998607781
    # huber_pca_runner4 = PCAPipelineRunner('Raw -> HuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug, n_components=100)
    # run_cv(huber_pca_runner4)

    # TODO: change alpha
    # huber_pca_runner2 = PCAPipelineRunner('Raw -> HuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=1.0)), beat_rnaseq, beat_drug)
    # run_cv(huber_pca_runner2)

    # huber_raw_runner = RawPipelineRunner('Raw -> HuberRegressor',
    #     MultiOutputRegressor(HuberRegressor(max_iter=10000, alpha=0.3)), beat_rnaseq, beat_drug)
    # run_cv(huber_raw_runner)

    # lasso_runner = PCAPipelineRunner('PCA -> MultiTaskLasso',
    #     MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    # run_cv(lasso_runner)

    # [-0.39381233 -0.41939928 -0.41497491 -0.59769095 -0.33431695], mean=-0.4320388857736034
    # lasso_runner2 = RawPipelineRunner('Raw MultiTaskLasso',
    #     MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    # run_cv(lasso_runner2)

    lasso_runner2 = RawPipelineRunner('Raw MultiTaskLasso',
        MultiTaskLasso(random_state=10, max_iter=10000, alpha=1.0), beat_rnaseq, beat_drug)
    run_subbmission2(lasso_runner2, subbmission2_folds)

    # 0.5: [-0.41743254 -0.42972947 -0.43007486 -0.58145634 -0.34809908], mean=-0.441358456207625
    # 0.7: [-0.39974979 -0.42063227 -0.41655715 -0.58168206 -0.33622506], mean=-0.4309692648978981
    # 0.8: [-0.39567269 -0.41916701 -0.41502366 -0.58667774 -0.3336461 ], mean=-0.43003744196637433
    # lasso_runner2 = RawPipelineRunner('Raw MultiTaskLasso',
    #     MultiTaskLasso(random_state=10, max_iter=10000, alpha=0.8), beat_rnaseq, beat_drug)
    # run_cv(lasso_runner2)

    # [-0.43818381 -0.44449487 -0.43987984 -0.64301492 -0.35742193], mean=-0.464599073746947
    # ridge = PCAPipelineRunner('PCA MultiOutputRegressor Ridge',
    #     MultiOutputRegressor(Ridge(random_state=10, max_iter=10000, alpha=1)), beat_rnaseq, beat_drug)
    # run_cv(ridge)

    # training_model = GradientBoostingRegressor()
    # spearman_runner = SpearmanCorrelationPipelineRunner(training_model, beat_rnaseq, beat_drug)
    # spearman_runner_results = spearman_runner.run_cross_validation(cv=5)
    # print("SpearmanCorr -> GradientBoostingRegressor results ")
    # print(f"{spearman_runner_results['test_score']}, mean={np.mean(spearman_runner_results['test_score'])}")

    # feature_selection_model = DecisionTreeRegressor()
    # dicision_tree_feature_selection = ModelFeatureSlectionPipelineRunner('DecisionTree -> GradientBoostingRegressor',
    #     training_model, feature_selection_model, beat_rnaseq, beat_drug)
    # run_cv(dicision_tree_feature_selection)

    # linear_regression_runner = PCAPipelineRunner('PCA -> LinearRegression',
    #     MultiOutputRegressor(LinearRegression()), beat_rnaseq, beat_drug)
    # run_cv(linear_regression_runner)

    # [-0.4060206  -0.44308826 -0.42111318 -0.59801123 -0.33965524], mean=-0.4415777023661193
    regressor_chain_runner = PCAPipelineRunner('PCA -> RegressorChain',
        RegressorChain(Lasso(alpha=0.7), order='random', random_state=42), beat_rnaseq, beat_drug, n_components=40)
    run_cv(regressor_chain_runner)

    # [-0.41126093 -0.4518314  -0.45571602 -0.65964898 -0.39580831], mean=-0.47485312811184155
    regressor_chain_runner2 = RawPipelineRunner('Raw RegressorChain',
        RegressorChain(Lasso(alpha=1.0), order='random', random_state=10), beat_rnaseq, beat_drug)
    run_cv(regressor_chain_runner2)


    # [-0.42703164 -0.48505265 -0.47001076 -0.64459804 -0.37438933], mean=-0.48021648398811057
    # runner = PCAPipelineRunner('PCA -> GradientBoostingRegressor',
    #     MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50)
    # run_cv(runner)

    # [-0.42092993 -0.45726059 -0.45008893 -0.60637066 -0.37943296], mean=-0.46281661457478795
    # raw_runner = RawPipelineRunner('Raw -> GradientBoostingRegressor',
    #     MultiOutputRegressor(GradientBoostingRegressor(random_state=42, max_features='log2')), beat_rnaseq, beat_drug)
    # run_cv(raw_runner)

    # [-0.40763929 -0.45144285 -0.45252828 -0.65356067 -0.35385169], mean=-0.46380455586758407
    # runner = PCAPipelineRunner('PCA -> RandomForestRegressor',
    #     MultiOutputRegressor(RandomForestRegressor(random_state=42)), beat_rnaseq, beat_drug, n_components=50)
    # run_cv(runner)

if __name__ == '__main__':
    main()