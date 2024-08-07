# sys.path.append("/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/07_transfer_benchmarking")

import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial
import scipy
import numpy as np
import anndata as ad
import os
import re
import pickle
import marker_transfer_helpers as th
import multiprocessing as mp
import sys


import knn_pairwise_methods as pknn


def get_markers_pos_neg(group1, group2, markers_dir_path, n_features_directional = 10, df_key_name="ensemble_gene_name", sort_by_col = "em_distance"):
    marker_files_present = os.listdir(markers_dir_path)
    # check for the names of the h0 and ha groups in the marker files
    # find the file that contains the h0 group name in its path name

    # Removed _ before name because marker file named became simplified
    # matching_files_group1 = [file for file in marker_files_present if re.search("_" + group1 + r'\.pkl', file)]
    matching_files_group1 = [file for file in marker_files_present if re.search(r'^'+ group1 + r'\.pkl', file)]
    if len(matching_files_group1) == 0:
        raise Exception(f"No marker files found for {group1}")
    if len(matching_files_group1) > 1:
        raise Exception(f"Multiple marker files found for {group1}")

    # Removed _ before name because marker file named became simplified
    # matching_files_group2 = [file for file in marker_files_present if re.search("_" + group2 + r'\.pkl', file)]
    matching_files_group2 = [file for file in marker_files_present if re.search(r'^'+ group2 + r'\.pkl', file)]
    if len(matching_files_group2) == 0:
        raise Exception(f"No marker files found for {group2}")
    if len(matching_files_group2) > 1:
        raise Exception(f"Multiple marker files found for {group2}")

    h0_obj_path = os.path.join(markers_dir_path, matching_files_group1[0])
    ha_obj_path = os.path.join(markers_dir_path, matching_files_group2[0])

    # print(f"loading marker objects from {h0_obj_path} and {ha_obj_path}")
    # load in the marker objects
    markers_obj_group1 = pickle.load(open(h0_obj_path, "rb"))
    markers_obj_group2 = pickle.load(open(ha_obj_path, "rb"))

    group_1_to2_df = markers_obj_group1[group2]
    group_2_to1_df = markers_obj_group2[group1]

    group_1_pos_markers_df = group_1_to2_df.sort_values(by=sort_by_col, ascending=False).head(n_features_directional)
    group_1_neg_markers_df = group_2_to1_df.sort_values(by=sort_by_col, ascending=False).head(n_features_directional)

    if (df_key_name not in group_1_pos_markers_df.columns) and (df_key_name not in group_1_neg_markers_df.columns):
        pos_genes = group_1_pos_markers_df.index.values
        neg_genes = group_1_neg_markers_df.index.values
    else:
        assert df_key_name in group_1_pos_markers_df.columns, "gene_name column not present in group 1 positive markers"
        assert df_key_name in group_1_neg_markers_df.columns, "gene_name column not present in group 1 negative markers"

        pos_genes = group_1_pos_markers_df[df_key_name]
        neg_genes = group_1_neg_markers_df[df_key_name]

    
 
    #ensure there is no overlap between the positive and negative genes
    assert len(set(pos_genes).intersection(set(neg_genes))) == 0 , "overlap between positive and negative genes"
    #combine the positive and negative genes and return result
    gene_set = list(set(pos_genes).union(set(neg_genes)))
    return gene_set

def compute_cosine_distance_test_statistic(h0_adata, ha_adata, query_adata, gene_set, n_sample=1000, test = "ks"):
    #set seed for sampling
    query_dense_mat = query_adata[:,gene_set].X.toarray()
    
    np.random.seed(0)
    h0_sampled_indices = np.random.choice(h0_adata.shape[0], n_sample, replace=True)
    ha_sampled_indices = np.random.choice(ha_adata.shape[0], n_sample, replace=True)

    h0_dense_mat = h0_adata[:,gene_set].X.toarray()
    ha_dense_mat = ha_adata[:,gene_set].X.toarray()

    h0_dense_mat = h0_dense_mat[h0_sampled_indices,:]
    ha_dense_mat = ha_dense_mat[ha_sampled_indices,:]

    #identify cells in each matrix that are all 0
    h0_all_zero_cells = np.where(np.sum(h0_dense_mat, axis=1)==0)[0]
    ha_all_zero_cells = np.where(np.sum(ha_dense_mat, axis=1)==0)[0]
    query_all_zero_cells = np.where(np.sum(query_dense_mat, axis=1)==0)[0]

    # if len(h0_all_zero_cells) > 0:
    #     print(f"h0 mat has all zeros at cells {h0_all_zero_cells}")

    # if len(ha_all_zero_cells) > 0:
    #     print(f"ha mat has all zeros at cells {ha_all_zero_cells}")

    # if len(query_all_zero_cells) > 0:
    #     print(f"query mat has all zeros at cells {query_all_zero_cells}")

    cosine_distances_h0 = spatial.distance.cdist(query_dense_mat, h0_dense_mat, metric='cosine')
    cosine_distances_ha = spatial.distance.cdist(query_dense_mat, ha_dense_mat, metric='cosine')

    #set distance of null values to 1
    cosine_distances_h0[np.isnan(cosine_distances_h0)] = 1
    cosine_distances_ha[np.isnan(cosine_distances_ha)] = 1

    #compute komogorov smirnov statistic for each row of cosine distances between query and h0 and ha

    assert cosine_distances_h0.shape == cosine_distances_ha.shape


    p_vals = []
    test_stats = []
    for cell_inx in range(cosine_distances_h0.shape[0]):
        vec_h0 = cosine_distances_h0[cell_inx,:]
        vec_ha = cosine_distances_ha[cell_inx,:]

        if test == "ks":
            test_stat, p_val = stats.ks_2samp(vec_h0, vec_ha, alternative='less')
        elif test == "t":
            test_stat, p_val = stats.ttest_ind(vec_h0, vec_ha, alternative='greater', equal_var=False)


        p_vals.append(p_val)
        test_stats.append(test_stat)


    result_df = pd.DataFrame({'p_val':p_vals, 'test_stat':test_stats, "query_cell_indices" :query_adata.obs.index.values, "test": test})
    return result_df



def compute_correlation_binary_to_nonzero(cellular_expression, comparison_profile):
    binary_cell_expression = cellular_expression > 0
    if np.sum(binary_cell_expression) == 0:
        return 0
    if np.sum(binary_cell_expression) == len(binary_cell_expression):
        return 0 
    correlation_val = np.corrcoef(binary_cell_expression, comparison_profile)
    return correlation_val[0,1]
# For each cell type, compute correlation of each cell in that group against the reference


def compute_correlations_on_group(query_obj, comparison_group, genes, reference_profile_obj, comparison_function=None):
    """
    Compute correlations on a specific group of cells against a reference.

    Parameters:
        reference_obj (anndata.AnnData): The reference object containing the cells being compared.
        genes (list): The list of genes to compute correlations for.
        group_name (str): The name of the group in the reference_object.
        comparison_group (str, optional): The name of the group construct a reference profile of and compare to.
        column_name (str): The column name in the reference object that contains the group information.
        reference_profile_obj (pd.DataFrame): The reference profile object containing the profiles to be correlated to.
        comparison_function (function, optional): The function to use for comparison. Defaults to None.

    Returns:
        list: The correlation values for each cell in the group.
    """

    if comparison_function is None:
        #error 
        print("No comparison function specified")
        return

    # Get list of genes for that cell type
    print(f"Getting comparison profile in reference profile object for {comparison_group}")
    nonzero_profile = reference_profile_obj.loc[comparison_group, genes]

    query_obj_target_genes = query_obj[:,genes]
    assert all(query_obj_target_genes.var_names == nonzero_profile.index)
    counts_mat = query_obj_target_genes.X.toarray()

    correlation_values = [comparison_function(counts_mat[i, :], nonzero_profile) for i in range(counts_mat.shape[0])]
    return correlation_values


def compute_distance_statistic_general(h0_adata, ha_adata, query_adata, gene_set, n_sample=1000, stat_test = "ranksum", distance_metric = "cosine", binarize = False):

    available_stats_tests = ["ks", "t", "ranksum"]
    available_distance_metrics = ["cosine", "correlation", "euclidean"]
    assert stat_test in available_stats_tests
    assert distance_metric in available_distance_metrics

    # make query data dense
    query_dense_mat = query_adata[:,gene_set].X.toarray()
    
    # take random sampling from the two groups being compared against the query
    np.random.seed(0)
    h0_sampled_indices = np.random.choice(h0_adata.shape[0], n_sample, replace=True)
    ha_sampled_indices = np.random.choice(ha_adata.shape[0], n_sample, replace=True)

    h0_dense_mat = h0_adata[:,gene_set].X.toarray()
    ha_dense_mat = ha_adata[:,gene_set].X.toarray()

    h0_dense_mat = h0_dense_mat[h0_sampled_indices,:]
    ha_dense_mat = ha_dense_mat[ha_sampled_indices,:]

    #identify cells in each matrix that are all 0
    h0_all_zero_cells = np.where(np.sum(h0_dense_mat, axis=1)==0)[0]
    ha_all_zero_cells = np.where(np.sum(ha_dense_mat, axis=1)==0)[0]
    query_all_zero_cells = np.where(np.sum(query_dense_mat, axis=1)==0)[0]

    # if len(h0_all_zero_cells) > 0:
    #     print(f"h0 mat has all zeros at cells {h0_all_zero_cells}")

    # if len(ha_all_zero_cells) > 0:
    #     print(f"ha mat has all zeros at cells {ha_all_zero_cells}")

    # if len(query_all_zero_cells) > 0:
    #     print(f"query mat has all zeros at cells {query_all_zero_cells}")

    # compute distance of query data against null and alternative distributions
    cosine_distances_h0 = spatial.distance.cdist(query_dense_mat, h0_dense_mat, metric=distance_metric)
    cosine_distances_ha = spatial.distance.cdist(query_dense_mat, ha_dense_mat, metric=distance_metric)

    #set distance of null values to 1
    cosine_distances_h0[np.isnan(cosine_distances_h0)] = 1
    cosine_distances_ha[np.isnan(cosine_distances_ha)] = 1

    #compute komogorov smirnov statistic for each row of cosine distances between query and h0 and ha

    assert cosine_distances_h0.shape == cosine_distances_ha.shape

    # for each cell determine if distance to null is stat smaller than alt.
    p_vals = []
    test_stats = []
    for cell_inx in range(cosine_distances_h0.shape[0]):
        vec_h0 = cosine_distances_h0[cell_inx,:]
        vec_ha = cosine_distances_ha[cell_inx,:]

        if stat_test == "ks":
            test_stat, p_val = stats.ks_2samp(vec_h0, vec_ha, alternative='less')
        elif stat_test == "t":
            test_stat, p_val = stats.ttest_ind(vec_h0, vec_ha, alternative='less', equal_var=False)
        elif stat_test == "ranksum":
            test_stat, p_val = stats.ranksums(vec_h0, vec_ha, alternative='less')

        p_vals.append(p_val)
        test_stats.append(test_stat)


    result_df = pd.DataFrame({'p_val':p_vals, 'test_stat':test_stats, "query_cell_indices" :query_adata.obs.index.values, "stat_test":stat_test, "distance_metric":distance_metric, "binarize":binarize, "n_sample":n_sample})
    return result_df

def transform_results_to_df(results_dict):
    results_df = pd.DataFrame()
    for compare_name, compare_result in results_dict.items():
        p_vals = compare_result["p_val"].values
        cell_names = compare_result["query_cell_indices"].values
        compare_result_df = pd.DataFrame(p_vals, columns=[compare_name], index = cell_names)
        # column bind the results
        results_df = pd.concat([results_df, compare_result_df], axis=1)
    return results_df


def test_out_of_dist_prediction_same_markers(true_group_name, test_group_name, compare_names, markers_path, chunked_reference_base, n_features = 25, clade_df = None, is_test = False):
    """
    true_group_name: str The group which the cells truly belong to, or the group being held out.
    test_group_name: str The cells which are being proposed as the cell type label of the cells. The current null hypothesis
    compare_names: list<str> names of the groups which to compare the query cells to. 
    markers_path: str Path to the directory containing all the pairwise markers
    """

    if not os.path.exists(markers_path):
        print(f"Marker path: {markers_path} DNE")
        return

    stat_method = "ranksum"
    distance_metric = "cosine"

    query_data = th.read_leaf_obj(true_group_name, chunked_obj_base=chunked_reference_base)
    false_h0_data = th.read_leaf_obj(test_group_name, chunked_obj_base=chunked_reference_base)

    sub_results = {}

    if is_test:
        print("TEST!")
        compare_names = compare_names[:2]

    for inx, compare_name in enumerate(compare_names):
        print(f"{inx} / {len(compare_names)} - {compare_name}")
        sys.stdout.flush()
        sys.stderr.flush()
        gene_set = get_markers_pos_neg(test_group_name, compare_name, markers_path, n_features_directional=n_features)

        compare_group_obj = th.read_leaf_obj(compare_name, chunked_obj_base=chunked_reference_base)
        wrong_pred_result = compute_distance_statistic_general(h0_adata=false_h0_data, ha_adata=compare_group_obj, query_adata=query_data, gene_set=gene_set, n_sample=1000, stat_test=stat_method, distance_metric=distance_metric)
        sub_results[compare_name] = wrong_pred_result

    final_results = {"true_group": true_group_name, "false_pred_group": test_group_name, "results": sub_results, "params": {"chunked_obj_base_reference": chunked_reference_base, "n_sample": 1000, "markers_path": markers_path, "n_features": n_features, "distance_metric": distance_metric, "stat_method": stat_method}}
    return final_results



def test_intra_clade_misplacements(true_group_name, full_compare_list, markers_path, chunked_reference_base, clade_df = None, n_features = 25, is_test = False):
    print(is_test)
    if clade_df is None:
        clade_df = th.load_clade_df()
    clade_name = clade_df.loc[true_group_name, "clade_name"]
    clade_members = clade_df[clade_df["clade_name"] == clade_name].index.values
    
    clade_members_non_identical = [member for member in clade_members if member != true_group_name]
    clade_members_non_identical = [leaf_node for leaf_node in clade_members_non_identical if leaf_node in full_compare_list]

    compare_names_non_identical = [name for name in full_compare_list if name != true_group_name]

    ood_result = {}

    n_clade_members = len(clade_members_non_identical)
    for inx, in_clade_leaf_name in enumerate(clade_members_non_identical):
        print(f"{inx}/{n_clade_members} - False Prediction: {in_clade_leaf_name}")
        sys.stdout.flush()
        sys.stderr.flush()

        non_predicted_groups = [group_name for group_name in compare_names_non_identical if group_name != in_clade_leaf_name]

        misplaced_result = test_out_of_dist_prediction_same_markers(true_group_name, in_clade_leaf_name, non_predicted_groups, markers_path, chunked_reference_base=chunked_reference_base, clade_df = clade_df, n_features=n_features, is_test = is_test)
        ood_result[in_clade_leaf_name] = misplaced_result

    return ood_result



def test_inter_clade_misplacements(true_group_name, full_compare_list, markers_path, chunked_reference_base, clade_df = None, n_features = 25, is_test = False, n_leaves = 10):
    """
    Function takes in current group being tested. It determines the clade and class that group is part of.
    It then exlcudes that clade but takes random leaf nodes from the same class and sets those as the false hypothesis leaving out the current group.

    """
    if clade_df is None:
        clade_df = th.load_clade_df()

    assert true_group_name in clade_df.index.values, f"True group {true_group_name} not in clade df"
    clade_name = clade_df.loc[true_group_name, "clade_name"]

    # group list not true group
    compare_names = [group for group in full_compare_list if group != true_group_name]

    # Get all the members of the same class
    class_name = true_group_name.split("_")[0]
    class_members = [group for  group in compare_names if group.split("_")[0] == class_name]
    print(f"Length of class members: {len(class_members)}")

    # Get all the members of the same clade
    leaf_clade_members = clade_df[clade_df["clade_name"] == clade_name].index.values
    print(f"Length of clade members: {len(leaf_clade_members)}")

    # get class members not in the clade
    class_members_not_in_clade = [group for group in class_members if group not in leaf_clade_members]
    n_class_members_not_in_clade = len(class_members_not_in_clade)
    print(f"Length of class members not in clade: {n_class_members_not_in_clade}")

    # if there exists leaves outside of clade, use them else use any random leaves outside of clade
    if n_class_members_not_in_clade == 0:
        # if none out of clade in class
        # select random members from out of class 
        not_in_clade_members = [group for group in compare_names if group not in leaf_clade_members]
        rand_number = np.min([n_leaves, len(not_in_clade_members)])
        np.random.seed(100)
        rand_members = np.random.choice(a = not_in_clade_members, size = rand_number, replace=False)
        print(f"Using random members from out of clade: {rand_members}")

    else:
        rand_number = np.min([n_leaves, len(class_members_not_in_clade)])
        np.random.seed(100)
        rand_members = np.random.choice(a = class_members_not_in_clade, size = rand_number, replace=False)

    ood_result = {}
    if is_test:
        rand_members = rand_members[:2]

    for inx, rand_member in enumerate(rand_members):
        print(f"{inx}/{len(rand_members)} - False Prediction: {rand_member}")
        sys.stdout.flush()
        sys.stderr.flush()

        non_predicted_groups = [group_name for group_name in compare_names if group_name != rand_member]

        misplaced_result = test_out_of_dist_prediction_same_markers(true_group_name = true_group_name, test_group_name = rand_member, compare_names = non_predicted_groups, markers_path = markers_path, chunked_reference_base=chunked_reference_base, clade_df = clade_df, n_features=n_features, is_test = is_test)
        ood_result[rand_member] = misplaced_result
    return ood_result



def read_h5py_element_to_pandas(h5py_element):
    # read h5py element to pandas
    import pandas as pd
    pandas_df = pd.DataFrame(h5py_element[:], columns=h5py_element.dtype.names)
    decoded_df = pandas_df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return pd.DataFrame(decoded_df)

def get_predictions_from_probs_mat(probs_mat, cell_names, cell_types):

    print(f"Prob Mat Shape: {probs_mat.shape}")

    #Ensure the shape matches the number of cells and expected cell types
    assert probs_mat.shape[0] == len(cell_names)
    assert probs_mat.shape[1] == len(cell_types)


    max_indices = np.argmax(probs_mat, axis=1)
    cell_type_predictions = [cell_types[i] for i in max_indices]
    prediction_df = pd.DataFrame({"cell.name": cell_names, "cell_type": cell_type_predictions})
    prediction_df.index = prediction_df["cell.name"]
    return prediction_df

def transform_results_to_df(results_dict):
    results_df = pd.DataFrame()
    for compare_name, compare_result in results_dict.items():
        p_vals = compare_result["p_val"].values
        cell_names = compare_result["query_cell_indices"].values
        compare_result_df = pd.DataFrame(p_vals, columns=[compare_name], index = cell_names)
        # column bind the results
        results_df = pd.concat([results_df, compare_result_df], axis=1)
    return results_df

def get_passing_df(results_dict, alpha = 0.05):
    results_df = transform_results_to_df(results_dict)
    passing_df = results_df < alpha
    return passing_df


def make_comparison_between_groups(query_adata, h0_group_name, ha_group_name,  markers_path, chunked_reference_path):

    stat_test = "ranksum"
    distance_metric = "cosine"

    gene_set = get_markers_pos_neg(group1 = h0_group_name, group2 = ha_group_name, markers_dir_path=markers_path)
    ha_adata = th.read_leaf_obj(leaf_name=ha_group_name, chunked_obj_base=chunked_reference_path)
    h0_adata = th.read_leaf_obj(leaf_name=h0_group_name, chunked_obj_base=chunked_reference_path)
    current_group_result = compute_distance_statistic_general(h0_adata = h0_adata, ha_adata = ha_adata, query_adata = query_adata, gene_set = gene_set, stat_test = stat_test, distance_metric = distance_metric)

    return current_group_result

def make_comparison_for_predicted_group(current_group_query, source_group, possible_groups_list, markers_path, chunked_reference_path):
    print(f"Making comparison for {source_group}")
    non_self_group = [g for g in possible_groups_list if g != source_group]
    source_group_result = {}
    for inx, compare_group in enumerate(non_self_group):
        current_group_result = make_comparison_between_groups(current_group_query, source_group, compare_group, markers_path, chunked_reference_path)
        source_group_result[compare_group] = current_group_result
    return source_group_result

def get_passing_cells_and_failing_results(compare_result, alpha = 0.05):
    failing_cell_results = {}

    res_df = transform_results_to_df(compare_result)
    n_cells = res_df.shape[0]

    passing_df = res_df < alpha
    for cell_inx in range(n_cells):
        failing_columns = passing_df.columns[~passing_df.iloc[cell_inx]]
        cell_name = res_df.index[cell_inx]
        failing_cell_results[cell_name] = failing_columns

    return failing_cell_results

def create_duplicated_counts_obj(current_result, query_adata, unique_index_marker="*_*"):
    sub_objects = []
    for cell_name, failing_groups in current_result.items():
        n_failing_groups = len(failing_groups)
        if n_failing_groups == 0:
            continue
        single_cell_obj = query_adata[cell_name, ]
        dense_counts_mat = single_cell_obj.X.todense()
        # duplicate counts mat n_failing_groups times
        duplicated_counts = np.tile(dense_counts_mat, (n_failing_groups, 1))

        new_index = [f"{cell_name}{unique_index_marker}{i}" for i in range(n_failing_groups)]
        obs_df = pd.DataFrame({"original_cell_name": [cell_name] * n_failing_groups, "predicted_cell_type": failing_groups}, index = new_index)
        duplicated_counts = ad.AnnData(X = duplicated_counts, var = single_cell_obj.var, obs = obs_df)
        duplicated_counts.X = scipy.sparse.csr_matrix(duplicated_counts.X)

        sub_objects.append(duplicated_counts)

    if len(sub_objects) == 0:
        return None

    combined_obj = ad.concat(sub_objects, axis=0)
    return combined_obj



def identify_cell_type_comparisons(prediction_df, min_p = 0):
#returns: dictionary of cell type comparisons to make
    n_cells = prediction_df.shape[0]
    cell_type_comparisons = {}
    for inx in range(n_cells):
        cell_name = prediction_df.index[inx]
        cell_type_probs = prediction_df.iloc[inx, :]
        max_cell_type = cell_type_probs.idxmax()
        predicted_conf_score = cell_type_probs[max_cell_type]
        has_prob_density_bool = cell_type_probs > min_p
        possible_cell_types = cell_type_probs.index[has_prob_density_bool]
        non_self_possible_cell_types = possible_cell_types[possible_cell_types != max_cell_type]

        res = {"prediction": max_cell_type, "cell_name": cell_name,"confidence": predicted_conf_score, "possible_cell_types": non_self_possible_cell_types}
        cell_type_comparisons[cell_name] = res
    
    return cell_type_comparisons



def predict_individual_cell_ensemble_models(cell_name, predicted_cell_type,
 cell_types_to_compare, expression_vector, model_path_base, inx = None, model_is_folder_chunked=False):
    #expression_vector is pandas series with gene names as index

    if inx is not None:
        if inx % 1000 == 0:
            print(inx)    

    prediction_results = {"prediction": predicted_cell_type, "cell_name": cell_name}
    cellular_comparison_list = []
    passes_hypothesis_global = True
    for compare_cell_type in cell_types_to_compare:
        #sort cell types to compare by lexicographic order
        if model_is_folder_chunked:
            knn_model_object = pknn.read_in_model(predicted_cell_type, compare_cell_type, model_path_base, suffix="classifier")
        else:
            model_name = "_".join(sorted([predicted_cell_type, compare_cell_type]))
            model_path = os.path.join(model_path_base, model_name + ".pkl")
            with open(model_path, "rb") as f:
                knn_model_object = pickle.load(f)
        markers = knn_model_object["markers"]
        model = knn_model_object["classifier"]


        # if a marker not not in the expression vector, add and set it to 0
        missing_markers = [marker for marker in markers if marker not in expression_vector.index]
        for marker in missing_markers:
            expression_vector[marker] = 0

        class_labels = model.classes_
        is_current_hypothesis = class_labels == predicted_cell_type
        #get expression vector for cell
        cell_expression = np.array(expression_vector[markers]).reshape(1, -1)
        #predict cell type
        cell_type_prediction_confidence = model.predict_proba(cell_expression)
        hypothesis_confidence = cell_type_prediction_confidence[0, is_current_hypothesis][0]
        passes_hypothesis = hypothesis_confidence >= 0.5

        if not passes_hypothesis:
                passes_hypothesis_global = False

        current_compare = {"comparison_group": compare_cell_type, "passes_hypothesis": passes_hypothesis, "confidence": hypothesis_confidence}
        cellular_comparison_list.append(current_compare)
    prediction_results["passes_hypothesis_cell"] = passes_hypothesis_global
    prediction_results["cellular_comparisons"] = cellular_comparison_list

    return prediction_results



def run_knn_ensemble_comparisons_parallel(cell_type_comparisons, model_path, query_obj, n_cores = 30, models_are_folder_chunked=False):
    assert len(list(cell_type_comparisons.keys())) == query_obj.shape[0]

    cell_names = list(query_obj.obs.index)
    gene_label_vector = query_obj.var.index

    # Converting query to dense to speed up arg construction
    print("Converting query to dense. Will spike memory.")
    query_counts_mat_dense = query_obj.X.toarray()
    #construct arguments for parallel processing
    print("Constructing args")
    args = [(cell_name, cell_type_comparisons[cell_name]["prediction"],
             cell_type_comparisons[cell_name]["possible_cell_types"], 
             pd.Series(query_counts_mat_dense[inx, :], index = gene_label_vector), 
             model_path, inx, models_are_folder_chunked) for inx, cell_name in enumerate(cell_names)]

    print('Beginning parallel processing')
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(predict_individual_cell_ensemble_models, args)

    results_dict = {cell_name: res for cell_name, res in zip(cell_names, results)}

    return results_dict

# identify failed cell type comparisosn and save a new set of the cell_name, predicted group (now being the group failed), and the types to compare to.
# Then run the knn_enemble_comparisons on the failed cells setting this new group to the default hypothesis
def identify_failed_cell_type_comparisons(knn_ensemble_results):
# returns: dictionary of cell type comparisons to make
#ex.) res = {"prediction": cell_type_hypothesis, "cell_name": cell_name, "possible_cell_types": non_self_possible_cell_types}
    cell_type_comparisons = []
    for inx, (cell_name, compare_results) in enumerate(knn_ensemble_results.items()):
        if not compare_results["passes_hypothesis_cell"]:
            results = compare_results["cellular_comparisons"]
            cell_type_passing_status_dict = {res["comparison_group"]: res["passes_hypothesis"] for res in results}
            cell_types_compared_to_list = list(cell_type_passing_status_dict.keys())
            for cell_type_compare in cell_types_compared_to_list:
                if not cell_type_passing_status_dict[cell_type_compare]:
                    new_cell_type_comparisons = [cell_type for cell_type in cell_types_compared_to_list if cell_type != cell_type_compare]
                    new_cell_type_comparisons.append(compare_results["prediction"])
                    res = {"prediction": cell_type_compare, "cell_name": cell_name, "possible_cell_types": new_cell_type_comparisons}
                    cell_type_comparisons.append(res)
                    # current_cell_results.append(res)
            # cell_type_comparisons[cell_name] = current_cell_results
    return cell_type_comparisons
    


def run_knn_ensemble_comparisons_from_cell_type_comparisons(cell_type_comparisons_failed,  query_obj, model_path, n_cores = 30, models_are_folder_chunked=False):
# construct arguments
    gene_label_vector = query_obj.var.index
    n_cell_type_comparisons_needed = len(cell_type_comparisons_failed)
    print(f"Constructing args for {n_cell_type_comparisons_needed} cell type comparisons")


    args = [(ele["cell_name"], ele["prediction"], ele["possible_cell_types"], pd.Series(query_obj[ele["cell_name"], :].X.toarray().flatten(), index=gene_label_vector), model_path, inx, models_are_folder_chunked) for inx, ele in enumerate(cell_type_comparisons_failed)]
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(predict_individual_cell_ensemble_models, args)
    return results

def construct_ensemble_model_args_parallel(cell_type_comparisons, query_obj, model_path, n_cores = 30):
    gene_label_vector = query_obj.var.index
    cell_names = list(query_obj.obs.index)
    with mp.Pool(n_cores) as pool:
        args = pool.starmap(construct_singular_arg_ele, [(cell_name, cell_type_comparisons, query_obj, model_path, inx) for inx, cell_name in enumerate(cell_names)])
    return args

def construct_singular_arg_ele(cell_name, cell_type_comparisons, query_obj, model_path, inx):
    gene_label_vector = query_obj.var.index
    return (cell_name, cell_type_comparisons[cell_name]["prediction"], cell_type_comparisons[cell_name]["possible_cell_types"], pd.Series(query_obj.X[inx,:].toarray().flatten(), index = gene_label_vector), model_path, inx)
