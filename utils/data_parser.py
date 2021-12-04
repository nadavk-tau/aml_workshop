import pandas as pd
import numpy as np
import enum
import os

from config import path_consts


class DataTransformation(object):
    # From project_guidelines.pdf:
    # We advise you to transform the gene expression data as follows: x -> log2(1+x), and to transform the
    # IC50 values: x -> log10(x) (but this is not mandatory if this does not improve performance)
    @staticmethod
    def log10(x):
        return np.log10(x)
    
    @staticmethod
    def log2(x):
        return np.log2(x + 1)


class ResourcesPath(enum.Enum):
    TCGA_RNA=0
    TCGA_MUT=1
    DRUG_MUT_COR=2
    DRUG_MUT_COR_LABELS=3
    BEAT_RNASEQ=4
    BEAT_DRUG=5
    BEAT_FOLDS=6

    def get_path(self):
        return os.path.join(path_consts.DATA_FOLDER_PATH, self.name.lower())

    def get_dataframe(self, should_replace_na=False, tranformation=None):
        df = pd.read_csv(self.get_path(), sep='\t')
        # NOTE: sklearn input format is:
        #       feature1 feature2 feature3 ...
        # item1
        # item2
        # item3
        # Therefore we need to transpose after reading the csv
        df = df.T
        if should_replace_na:
            # TODO: try different impls here
            df = df.fillna(df.mean())
        if tranformation:
            df = tranformation(df)
        return df


class SubmissionFolds(object):
    @staticmethod
    def get_submission2_beat_folds():
        folds = pd.read_csv(ResourcesPath.BEAT_FOLDS.get_path(), sep='\t', names=['patient', 'fold'])
        result = []
        for fold_index, fold_group in folds.groupby('fold'):
            train = folds[folds.fold != fold_index].index.values
            test = fold_group.index.values
            result.append((train, test))
        return result
