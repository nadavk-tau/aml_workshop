import pathlib

# First '.parent' for 'config' dir, second for the project's root path
PROJECT_ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_FOLDER_PATH = pathlib.Path(PROJECT_ROOT_PATH, r'data')
RESULTS_FOLDER_PATH = pathlib.Path(PROJECT_ROOT_PATH, r'results')
TASK3_FEATURES_PATH = pathlib.Path(PROJECT_ROOT_PATH, r'data/task3_features')
TRAINNED_MDOELS_PATH = pathlib.Path(PROJECT_ROOT_PATH, r'trained_models')
FINAL_TRAINNED_MDOELS_PATH = pathlib.Path(PROJECT_ROOT_PATH, r'trained_models/final_model')
