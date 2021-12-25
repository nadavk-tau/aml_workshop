
import pandas as pd

from scipy.stats import spearmanr
from utils.data_parser import ResourcesPath


def corr_function_generator(corr_function):
    def corr_wrapper(X, y):
        scores = []
        p_values = []

        mutant_group_indeies = (y == True)
        no_mutant_indeies = (y == False)

        for col in range(X.shape[1]):
            try:
                mutant_group = X[mutant_group_indeies, col]
                no_mutant_group = X[no_mutant_indeies, col]

                score, p_value = corr_function(mutant_group, no_mutant_group)
            except ValueError:
                score = 0
                p_value = 1

            scores.append(score)
            p_values.append(p_value)

        return scores, p_value

    return corr_wrapper

def calculate_mutation_drug_correlation_matrix(mutation_matrix, drug_respose):
    results = {mutation: [] for mutation in mutation_matrix.columns}
    for mutation in mutation_matrix.columns:
        mutations_vector = mutation_matrix[mutation]
        mutations_indeies = set(mutations_vector.index)

        for drug in drug_respose.columns:
            drug_response_vector = drug_respose[drug].dropna()
            intersecting_indeies = set(drug_response_vector.index) & mutations_indeies

            mut_vec = mutations_vector.loc[intersecting_indeies]
            drug_vec = drug_response_vector.loc[intersecting_indeies]

            corr, p_value = spearmanr(mut_vec, drug_vec)
            results[mutation].append(corr)

    drug_mutation_corr_matrix = pd.DataFrame.from_dict(results)
    drug_mutation_corr_matrix.index = drug_respose.columns
    drug_mutation_corr_matrix = drug_mutation_corr_matrix.fillna(0)

    return drug_mutation_corr_matrix