import pandas as pd
import gc

from sklearn.cluster import KMeans
from scipy.stats import spearmanr


def get_top_50_featuers_for_set(X, drugs):
    print("get_top_50_featuers_for_set")
    selected_genes = set()

    for drug in drugs.columns:
        y = drugs[drug]
        features = []
        for gene_index, gene in enumerate(X.T):
            corr, _ = spearmanr(gene, y)
            features.append((gene_index, abs(corr)))

        gc.collect()
        current_top_featuers = sorted(features, key=lambda x: x[1], reverse=True)[:50]
        selected_genes = selected_genes | set([gene_index for gene_index, corr in current_top_featuers])

    return list(selected_genes)


def calc_spearman_for_selected_genes(X, drugs, selected_genes):
    drug_to_selected_genes_corr = {}

    for drug in drugs.columns:
        y = drugs[drug]
        drug_gene_correnlation = []
        for gene in selected_genes:
            corr, _ = spearmanr(X.T[gene], y)
            drug_gene_correnlation.append(corr)
        drug_to_selected_genes_corr[drug] = drug_gene_correnlation

    return pd.DataFrame.from_dict(drug_to_selected_genes_corr, orient='index', columns=selected_genes)


def cluster_drugs(X, drugs, clusters_number=3):
    print("cluster_drugs")
    selected_genes = get_top_50_featuers_for_set(X, drugs)
    print("got top 50")
    spearman_corr_matrix = calc_spearman_for_selected_genes(X, drugs, selected_genes)
    print("Calculated spearman corr matrix")

    kmeans_model = KMeans(n_clusters=clusters_number)
    kmeans_model.fit(spearman_corr_matrix)
    clusters = kmeans_model.predict(spearman_corr_matrix)

    return clusters