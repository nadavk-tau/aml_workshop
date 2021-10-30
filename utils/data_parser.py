import pandas as pd
import numpy as np
import pathlib
import enum
import seaborn as sns

from config import path_consts


class ResourcesPath(enum.Enum):
    TCGA_RNA=0
    TCGA_MUT=1
    DRUG_MUT_COR=2
    DRUG_MUT_COR_LABELS=3
    BEAT_RNASEQ=4
    BEAT_DRUG=5

    def get_path(self):
        return path_consts.DATA_FOLDER_PATH / self.name.lower()

    def get_dataframe(self, should_replace_na=False, should_log_tranform=False):
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
        if should_log_tranform:
            df = np.log(df)

        return df.copy()


