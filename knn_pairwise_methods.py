# sys.path.append("/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/06_marker_based_transfer/02_marker_based_transfer_methods")


from sklearn.neighbors import KNeighborsClassifier
import anndata as ad
import pandas as pd
import numpy as np
import multiprocessing as mp
import scanpy as sc
import sys
import time
import os
import pickle
import comparison_functions as cf
import marker_transfer_helpers as th
import gc

def create_cellular_predictions_and_comparisons_n_groups(combined_obj, k_for_top_ct = 10, n_neighbor_cts_needed = 5, k_radius_search = 500, embed_key = "X_pca", ref_ct_label = "ClusterNm"):
    # n_neighbors_needed = n_neighbor_cts_needed * max_size_per_group
    ref_data = combined_obj[combined_obj.obs["source"] == "reference"]
    ref_coords=ref_data.obsm[embed_key]
    ref_labels = ref_data.obs[ref_ct_label]

    # dont't know if n_neighbors is needed
    print(f"Building knn graph for reference data.")
    ref_knn_classifier = KNeighborsClassifier(n_neighbors=10)
    ref_knn_classifier.fit(X = ref_coords, y = ref_labels)

    print(f"Running knn classifier for query data. k = {k_radius_search}")
    query_data = combined_obj[combined_obj.obs["source"] == "query"]
    query_coords=query_data.obsm[embed_key]
    query_result_mats = ref_knn_classifier.kneighbors(query_coords, n_neighbors = k_radius_search, return_distance=True)

    cell_indices_mat = query_result_mats[1]
    query_cell_names = query_data.obs_names
    parse_comparison_results_args = [(i, query_cell_names[i], cell_indices_mat[i, ], ref_labels, k_for_top_ct, n_neighbor_cts_needed) for i in range(cell_indices_mat.shape[0])]

    print("Parsing knn results. Establish future comparisons")
    with mp.Pool(30) as pool:
        results = pool.starmap(parse_knn_for_hypothesis_and_comparisons, parse_comparison_results_args)

    results_dict = {ele["cell_name"]: ele for ele in results}

    return results_dict



def parse_knn_for_hypothesis_and_comparisons(inx, cell_name, cell_indices_vec, full_ref_labels_vec, k_for_top_ct = 10, n_neighbor_cts_needed = 5):
    if inx % 1000 == 0:
        print(inx)

    assert max(cell_indices_vec) < len(full_ref_labels_vec), "Cell indices are out of range"
    #ensure full_ref_labels_vec is a pandas series
    if not isinstance(full_ref_labels_vec, pd.Series):
        full_ref_labels_vec = pd.Series(full_ref_labels_vec)

    #get cell types for the indices interest in for query
    ref_labels_vec = full_ref_labels_vec[cell_indices_vec]
    # order the cell types for top k by frequency
    top_k_cts = ref_labels_vec[:k_for_top_ct].value_counts()
    top_ct_names = top_k_cts.index
    # get top cell type and its p for top k
    top_ct = top_ct_names[0]
    top_ct_prop = top_k_cts[0] / k_for_top_ct

    cts_non_top = ref_labels_vec[ref_labels_vec != top_ct]
    _, first_seen_inx = np.unique(cts_non_top, return_index=True)
    sorted_first_seen_inx=np.sort(first_seen_inx)
    first_seen_non_top_cts = cts_non_top[sorted_first_seen_inx]
    first_seen_non_self_cts = first_seen_non_top_cts[:n_neighbor_cts_needed].tolist()

    res = {"prediction": top_ct, "cell_name": cell_name, "confidence": top_ct_prop, "possible_cell_types": first_seen_non_self_cts}
    return res



def parse_combined_results_obj(combined_obj):
    cellular_results = {}
    for inx, ele in enumerate(combined_obj):
        if inx % 1000 == 0:
            print(inx)
        cell_name = ele["cell_name"]
        prediction = ele["prediction"]
        current_ele_lis = cellular_results.get(cell_name, [])
        passes_comparison = ele["passes_hypothesis_cell"]
        comparisons = ele["cellular_comparisons"]
        passing_list = np.array([res["passes_hypothesis"] for res in comparisons])
        if len(passing_list) == 0:
            passing_rate = 1
        else:
            passing_rate = np.mean(passing_list)
        result_tuple = (passing_rate, prediction, passes_comparison, comparisons)
        current_ele_lis.append(result_tuple)
        cellular_results[cell_name] = current_ele_lis


    # sort results
    cellular_results = {cell_name: sorted(res, key = lambda x: x[0], reverse=True) for cell_name, res in cellular_results.items()}


    cell_names = list(cellular_results.keys())
    cell_type_predictions = [cellular_results[cell_name][0][1] for cell_name in cell_names]
    pass_rate = [cellular_results[cell_name][0][0] for cell_name in cell_names]
    pred_df = pd.DataFrame({"cell_type_prediction": cell_type_predictions, "pass_rate": pass_rate}, index=cell_names)

    return cellular_results, pred_df


def preform_pairwise_knn_prediction( model_path, n_neighbors_h0, query_obj, ref_obj = None, combined_obj = None, n_next_cell_types_compare = None, n_cores = 30, embed_key="X_pca", ref_ct_label = "ClusterNm", models_are_folder_chunked=True):

    assert query_obj is not None, "query_obj must be provided"

    if combined_obj is None:
        if ref_obj is None:
            raise ValueError("Either combined_obj or query_obj and ref_obj must be provided")
        # combine query and ref objects
        query_obj.obs["source"] = "query"
        ref_obj.obs["source"] = "reference"
        combined_obj = ad.concat([query_obj, ref_obj])
        combined_obj = run_clustering_workflow(combined_obj)
    else:
        # subset combined_obj with query_source to only include query cells
        # get number of query observations
        n_query = np.sum(combined_obj.obs["source"] == "query")
        if n_query != query_obj.shape[0]:
            query_index_set = set(query_obj.obs.index)
            combined_inx_in_query = np.array([ele in query_index_set for ele in combined_obj.obs.index])
            is_query_and_not_in_query_obj = (combined_obj.obs["source"] == "query") & (~combined_inx_in_query)
            if np.sum(is_query_and_not_in_query_obj) > 0:
                print("Removing cells from combined_obj that are not in query_obj")
                print("Number of cells removed: ", np.sum(is_query_and_not_in_query_obj))
            combined_obj = combined_obj[~is_query_and_not_in_query_obj]
            if ref_obj is not None:
                raise ValueError("Either combined_obj or ref_obj must be provided, not both")

    # run knn prediction
    if n_next_cell_types_compare is None:
        # run pairwise comparisons
        print("Running pairwise comparisons for cellular predictions, n_next_cell_types_compare not provided")
        h0_predictions = preform_knn_prediction_from_combined(combined_obj, batch_var = "source", ref_value = "reference", ref_label = ref_ct_label, n_neighbors=n_neighbors_h0, embed_key = embed_key)
        cell_type_comparisons = cf.identify_cell_type_comparisons(h0_predictions)
    else:
        print("Running n_groups comparisons for cellular predictions. n_next_cell_types_compare provided.")
        h0_predictions = None
        # print the time
        print(time.ctime())
        cell_type_comparisons = create_cellular_predictions_and_comparisons_n_groups(combined_obj, k_for_top_ct = n_neighbors_h0, n_neighbor_cts_needed = n_next_cell_types_compare, embed_key=embed_key, ref_ct_label=ref_ct_label)

    gc.collect()
    # print time
    print(time.ctime())
    knn_ensemble_results = cf.run_knn_ensemble_comparisons_parallel(cell_type_comparisons, model_path, query_obj, n_cores = n_cores, models_are_folder_chunked=models_are_folder_chunked)
    print(time.ctime())
    print("Parsing first round results")
    gc.collect()
    cell_type_comparisons_failed = cf.identify_failed_cell_type_comparisons(knn_ensemble_results)
    print(time.ctime())
    gc.collect()
    print("Running secondary comparisons")
    seconary_results = cf.run_knn_ensemble_comparisons_from_cell_type_comparisons(cell_type_comparisons_failed, query_obj, model_path, n_cores = n_cores, models_are_folder_chunked=models_are_folder_chunked)
    gc.collect()
    print(time.ctime())
    print("Seconary results done")
    current_result_list = [val for val in knn_ensemble_results.values()]
    print(time.ctime())
    combined_result_obj = current_result_list + seconary_results
    print("Parsing combined results")
    cellular_results, cellular_prediction_df = parse_combined_results_obj(combined_result_obj)
    print(time.ctime())
    res = {"cellular_prediction_df": cellular_prediction_df, "h0_predictions": h0_predictions, "initial_ensemble_results": knn_ensemble_results, "cell_type_comparisons": cell_type_comparisons, "seconary_results": seconary_results, "combined_result_obj": combined_result_obj}
    return res

def run_clustering_workflow(combined_obj, markers = None):
    # run clustering workflow on combined object
    #Normalize and log transform
    print("Normalizing and log transforming")
    sc.pp.normalize_total(combined_obj, target_sum=1e4)
    sc.pp.log1p(combined_obj)

    print("Finding variable genes")
    # Find variable genes
    if markers is None:
        sc.pp.highly_variable_genes(combined_obj, n_top_genes=4000, flavor="seurat")
    else:
        assert all([marker in combined_obj.var.index for marker in markers]), "Not all markers are in the combined object"
        combined_obj.var["highly_variable"] = combined_obj.var.index.isin(markers)

    # Filter genes
    # filter to hvgs
    print("Filtering to highly variable genes")
    combined_obj = combined_obj[:, combined_obj.var["highly_variable"]].copy()
    print(combined_obj)


    # Scale data
    print("Scaling data")
    sc.pp.scale(combined_obj, max_value=10)

    print("Running PCA")
    # Run PCA
    sc.tl.pca(combined_obj, svd_solver='arpack', n_comps=80)
    return combined_obj

def preform_knn_prediction_from_combined(combined_obj, batch_var = "source", ref_value = "reference", ref_label = "ClusterNm", n_neighbors=10, embed_key = "X_pca"):
    # isolate embeddings
    ref_indices = combined_obj.obs[batch_var] == ref_value
    query_indices = combined_obj.obs[batch_var] != ref_value
    ref_embeddings = combined_obj.obsm[embed_key][ref_indices]
    query_embeddings = combined_obj.obsm[embed_key][query_indices]
    ref_labels = combined_obj.obs[ref_label][ref_indices]
    query_cell_names = combined_obj.obs_names[query_indices]

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(ref_embeddings, ref_labels)
    query_preds = knn_classifier.predict_proba(query_embeddings)
    leaf_ordering = knn_classifier.classes_
    prediction_data_frame = pd.DataFrame(query_preds, columns=leaf_ordering, index=query_cell_names)
    return prediction_data_frame

def preform_knn_prediction(query_embeddings, query_cell_names, ref_embeddings, ref_labels, n_neighbors=10):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(ref_embeddings, ref_labels)
    query_preds = knn_classifier.predict_proba(query_embeddings)
    leaf_ordering = knn_classifier.classes_
    prediction_data_frame = pd.DataFrame(query_preds, columns=leaf_ordering, index=query_cell_names)
    return prediction_data_frame




# Model Training
def create_classifier_pair(group1, group2, chunked_reference_base, markers, cell_per_type = 200, from_macosko=True, equal_size_n = False):
  
    group1_obj = th.read_leaf_obj(group1, chunked_reference_base, from_macosko=from_macosko)
    group2_obj = th.read_leaf_obj(group2, chunked_reference_base, from_macosko=from_macosko)

    # subset each object to the min of cell_per_type cells or the number of cells in the object
    n_cells_group_1 = group1_obj.shape[0]
    n_cells_group_2 = group2_obj.shape[0]


    if equal_size_n:
      min_size = np.min(n_cells_group_1, n_cells_group_2)
      if min_size < cell_per_type:
        n_to_sample_to = min_size
      else:
        n_to_sample_to = cell_per_type
      group1_obj =  group1_obj[np.random.choice(n_cells_group_1, n_to_sample_to, replace=False), :].copy()
      group2_obj =  group2_obj[np.random.choice(n_cells_group_2, n_to_sample_to, replace=False), :].copy()
    else:
      if n_cells_group_1 > cell_per_type:
          group1_obj = group1_obj[np.random.choice(n_cells_group_1, cell_per_type, replace=False), :].copy()
      if n_cells_group_2 > cell_per_type:
          group2_obj = group2_obj[np.random.choice(n_cells_group_2, cell_per_type, replace=False), :].copy()


    # print(group1_obj.shape)
    # print(group2_obj.shape)
  
    #normalize log both objects
    sc.pp.normalize_total(group1_obj, target_sum=1e4)
    sc.pp.log1p(group1_obj)

    sc.pp.normalize_total(group2_obj, target_sum=1e4)
    sc.pp.log1p(group2_obj)

    group1_mat_markers_only = group1_obj[:, markers].X.toarray()
    group2_mat_markers_only = group2_obj[:, markers].X.toarray()

    knn_classifer = KNeighborsClassifier(n_neighbors=12, metric="cosine", weights="distance")
    reference_labels = [group1] * group1_mat_markers_only.shape[0] + [group2] * group2_mat_markers_only.shape[0]
    training_data = np.vstack([group1_mat_markers_only, group2_mat_markers_only])
    knn_classifer.fit(training_data, reference_labels)
    return knn_classifer

def create_classifier_set(inx, current_cell_type, compare_cell_types, chunked_reference_base, markers_base, base_classifier_path, from_macosko=True, n_features_directional=25, cell_per_type=200, equal_size_n=False):
    f_name="classifier"
    if inx % 10 == 0:
        print(f"Working on inx: {inx}")
    # base_classifier_path = "/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/07_transfer_benchmarking/objects/whole_brain_classifiers"
    if not os.path.exists(base_classifier_path):
        os.makedirs(base_classifier_path)
    for cell_type in compare_cell_types:
        model_exists = check_if_model_exists(current_cell_type, cell_type, base_classifier_path,f_name)
        if model_exists:
            continue

        markers = cf.get_markers_pos_neg(current_cell_type, cell_type, markers_base, n_features_directional=n_features_directional, sort_by_col = "pct_diff")
        classifier = create_classifier_pair(group1 = current_cell_type, group2=cell_type, 
                                            chunked_reference_base=chunked_reference_base, 
                                            markers=markers, from_macosko=from_macosko, cell_per_type=cell_per_type, 
                                            equal_size_n=equal_size_n)
        res_obj = {"classifier": classifier, "cell_type1": cell_type, "cell_type2": current_cell_type,  "markers": markers}
        # save name as sorted pair of cell types
        # pickle.dump(res_obj, open(classifier_path, "wb"))
        save_pairwise_model(current_cell_type, cell_type, base_classifier_path, res_obj, f_name, order_alphabetically=True)


def save_pairwise_model(group1, group2, base_path, obj, suffix, order_alphabetically=True):
    # sort names alphabetically
    # make the outer folder the first name and inner the second
    if order_alphabetically:
        if group1 > group2:
            group1, group2 = group2, group1

    outer_folder = os.path.join(base_path, group1)
    if not os.path.exists(outer_folder):
        os.makedirs(outer_folder)

    inner_folder = os.path.join(outer_folder, group2)
    if not os.path.exists(inner_folder):
        os.makedirs(inner_folder)
    
    model_path = os.path.join(inner_folder, f"{suffix}.pkl")
    # print(f"saving model to {model_path}")
    pickle.dump(obj, open(model_path, "wb"))


def check_if_model_exists(group1, group2, base_path, suffix, sort_alphabetically=True):
    if sort_alphabetically:
        if group1 > group2:
            group1, group2 = group2, group1
    outer_folder = os.path.join(base_path, group1)
    if not os.path.exists(outer_folder):
        return False

    inner_folder = os.path.join(outer_folder, group2)
    if not os.path.exists(inner_folder):
        return False

    model_path = os.path.join(inner_folder, f"{suffix}.pkl")
    if not os.path.exists(model_path):
        return False
    return True

def read_in_model( group1, group2, base_path, suffix, sort_alphabetically=True):
    if sort_alphabetically:
        if group1 > group2:
            group1, group2 = group2, group1
    outer_folder = os.path.join(base_path, group1)
    inner_folder = os.path.join(outer_folder, group2)
    model_path = os.path.join(inner_folder, f"{suffix}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model