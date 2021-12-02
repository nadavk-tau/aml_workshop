import pandas as pd
import numpy as np
import enum

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
        return path_consts.DATA_FOLDER_PATH / self.name.lower()

    def get_dataframe(self, should_replace_na=False, tranformation=None):
        df = pd.read_csv(self.get_path(), sep='\t').copy()

        # NOTE: sklearn input format is:
        #       feature1 feature2 feature3 ...
        # item1
        # item2
        # item3
        # Therefore we need to transpose after reading the csv

        df = df.T
        if should_replace_na:
            df = df.fillna(df.mean())
        if tranformation:
            df = tranformation(df)

        return df.copy()
