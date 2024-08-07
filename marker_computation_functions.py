import pandas as pd
import scipy.stats as stats
import marker_transfer_helpers as cf
import sys

def compare_gene_two_groups_em(gene_name, group_1_obj, group_2_obj):
    group_1_gene_vec = group_1_obj[:, gene_name].X.toarray().flatten()
    group_2_gene_vec = group_2_obj[:, gene_name].X.toarray().flatten()
    em_distance = stats.wasserstein_distance(group_1_gene_vec, group_2_gene_vec)
    return em_distance

def identify_genes_to_compare(group_1, group_2, pct_nonzero_df, min_pct_diff = 0.3, min_reference_expression = 0.4, max_genes = 300):
    group_1_genes = pct_nonzero_df.loc[group_1]
    group_2_genes = pct_nonzero_df.loc[group_2]
    group_1_2_diff_values = group_1_genes - group_2_genes

    gene_df = pd.DataFrame({"gene_name": group_1_2_diff_values.index, "diff": group_1_2_diff_values.values, "source_group": group_1_genes.values}, index = group_1_2_diff_values.index)
    meets_criteria_bool = (gene_df["diff"] > min_pct_diff) & (gene_df["source_group"] > min_reference_expression)
    marker_genes_df = gene_df[meets_criteria_bool]
    marker_genes_df = marker_genes_df.sort_values(by="diff", ascending=False)
    if (marker_genes_df.shape[0] > max_genes):
        marker_genes_df = marker_genes_df.iloc[:max_genes, :]
    return(marker_genes_df)

def convert_ensemble_to_gene_name(ensemble_id, anndata_obj):
    var_df = anndata_obj.var
    assert ensemble_id in var_df.index, "Ensemble ID not in var df"
    assert "gene_name" in var_df.columns, "Gene name not in var df"

    gene_name = anndata_obj.var.loc[ensemble_id, "gene_name"]
    return(gene_name)

def get_max_difference_nonzero_genes_vector(group1, group2, pct_nonzero_df, n_genes = 30):
    assert group1 in pct_nonzero_df.index, "Group1 not in pct_nonzero_df"
    assert group2 in pct_nonzero_df.index, "Group2 not in pct_nonzero_df"

    group1_genes = pct_nonzero_df.loc[group1]
    group2_genes = pct_nonzero_df.loc[group2]
    group1_2_diff_values = group1_genes - group2_genes
    group1_2_diff_values = group1_2_diff_values.sort_values(ascending=False)

    return group1_2_diff_values[:n_genes]
    # gene_names = group1_2_diff_values.index[:n_genes]
    # return gene_names

def compute_em_markers(group_1_name, group_2_name, pct_nonzero_df, chunked_reference_base_path, n_genes = 30, group_resolution = None, group_1_obj = None):
    possble_resolutions = ["clade", "leaf", "class"]
    if group_resolution not in possble_resolutions:
        print("Group resolution not valid")
        sys.exit(1)

    if group_1_obj is None:
        if group_resolution == "clade":
            group_1_obj = cf.read_clade_obj(group_1_name, chunked_obj_base = chunked_reference_base_path)
        elif group_resolution == "leaf":
            group_1_obj = cf.read_leaf_obj(group_1_name, chunked_obj_base = chunked_reference_base_path)

        print(f"Loaded Group1: {group_1_name}")
    else:
        print(f"Using provided group1 object: {group_1_name}")


    if group_resolution == "clade":
        group_2_obj = cf.read_clade_obj(group_2_name, chunked_obj_base = chunked_reference_base_path)
    elif group_resolution == "leaf":
        group_2_obj = cf.read_leaf_obj(group_2_name, chunked_obj_base = chunked_reference_base_path)

    print(f"Loaded Group2: {group_2_name}")
    print(group_2_obj)


    diff_values_em = get_max_difference_nonzero_genes_vector(group_1_name, group_2_name, pct_nonzero_df, n_genes = n_genes)
    genes_for_em = diff_values_em.index.tolist()
    gene_candidates_df = pd.DataFrame({"ensemble_gene_name": genes_for_em, "nonzero_diff": diff_values_em.values}, index = genes_for_em)

    em_distance_results = []
    for gene_name in genes_for_em:
        em_distance_results.append(compare_gene_two_groups_em(gene_name, group_1_obj=group_1_obj, group_2_obj=group_2_obj))
    gene_candidates_df["em_distance"] = em_distance_results
    gene_candidates_df["gene_name"] = gene_candidates_df["ensemble_gene_name"].apply(lambda x: cf.convert_ensemble_to_gene_name(x))

    return gene_candidates_df