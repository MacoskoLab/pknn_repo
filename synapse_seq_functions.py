
import multiprocessing as mp
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import pickle
import os
import gc
import time

from sklearn.neighbors import KNeighborsClassifier

def compute_and_save_pct_diff_markers(base_chunked_dir, cell_type_1, cell_type_2, n_genes, out_dir_base):
    # load objs
    cell_type_1_path = os.path.join(base_chunked_dir, f"{cell_type_1}.h5ad")
    cell_type_2_path = os.path.join(base_chunked_dir, f"{cell_type_2}.h5ad")

    assert os.path.exists(cell_type_1_path), f"{cell_type_1_path} DNE"
    assert os.path.exists(cell_type_2_path), f"{cell_type_2_path} DNE"

    cell_type_1_adata = ad.read_h5ad(cell_type_1_path)
    cell_type_2_adata = ad.read_h5ad(cell_type_2_path)

    assert np.all(cell_type_1_adata.var_names == cell_type_2_adata.var_names)
    gene_names = cell_type_1_adata.var_names
    cell_type_1_nonzero_series = pd.Series(np.array(np.mean(cell_type_1_adata.X > 0, axis=0)).flatten(), index=gene_names)
    cell_type_2_nonzero_series = pd.Series(np.array(np.mean(cell_type_2_adata.X > 0, axis=0)).flatten(), index=gene_names)
    
    group_1_pos_diff = cell_type_1_nonzero_series - cell_type_2_nonzero_series
    group_1_pos_sorted = group_1_pos_diff.sort_values(ascending=False)
    group_1_markers = group_1_pos_sorted[:n_genes]
    
    group_2_pos_diff = cell_type_2_nonzero_series - cell_type_1_nonzero_series
    group_2_pos_sorted = group_2_pos_diff.sort_values(ascending=False)
    group_2_markers = group_2_pos_sorted[:n_genes]
    
    res = {cell_type_1: group_1_markers, cell_type_2: group_2_markers}

    save_pairwise_model(cell_type_1, cell_type_2, out_dir_base, res, "markers")
    
def read_in_sorted_subfolder_obj( group1, group2, base_path, suffix=None, sort_alphabetically=True):
    if sort_alphabetically:
        if group1 > group2:
            group1, group2 = group2, group1
    outer_folder = os.path.join(base_path, group1)
    inner_folder = os.path.join(outer_folder, group2)
    if suffix is not None:
        model_path = os.path.join(inner_folder, f"{suffix}.pkl")
    else:
        objs_present = os.listdir(inner_folder)
        if len(objs_present) != 1:
            raise ValueError("Must have one object or specify a suffix")
        fname = objs_present[0]
        model_path = os.path.join(inner_folder, fname)
            
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def get_markers_chunked(cell_type_1, cell_type_2, markers_base, suffix=None):

    obj = read_in_sorted_subfolder_obj(cell_type_1, cell_type_2, markers_base, suffix=suffix)
    markers_1 = obj[cell_type_1]
    markers_2 = obj[cell_type_2]
    
    combined_markers_index = markers_1.index.append(markers_2.index)
    # Convert to NumPy array
    combined_markers = np.array(combined_markers_index)
    return combined_markers


def create_classifier_set(inx, current_cell_type, compare_cell_types, chunked_reference_base, markers_base, base_classifier_path, n_features_directional=25, cell_per_type=200, equal_size_n=False):
    f_name="classifier"
    if inx % 10 == 0:
        print(f"Working on inx: {inx}")

    if not os.path.exists(base_classifier_path):
        os.makedirs(base_classifier_path)
    for cell_type in compare_cell_types:
        model_exists = check_if_model_exists(current_cell_type, cell_type, base_classifier_path, f_name)
        if model_exists:
            continue

        markers = get_markers_chunked(current_cell_type, cell_type, markers_base=markers_base, suffix="markers")
                                      
        classifier = create_classifier_pair(cell_type_1 = current_cell_type, cell_type_2=cell_type, 
                                            chunked_reference_base=chunked_reference_base, 
                                            markers=markers,  cell_per_type=cell_per_type, 
                                            equal_size_n=equal_size_n)
        
        res_obj = {"classifier": classifier, "cell_type1": cell_type, "cell_type2": current_cell_type,  "markers": markers}

        # save in indexable system
        save_pairwise_model(current_cell_type, cell_type, base_classifier_path, res_obj, f_name, order_alphabetically=True)



def create_classifier_pair(cell_type_1, cell_type_2, chunked_reference_base, markers, cell_per_type = 200, equal_size_n = False):
    
    cell_type_1_path = os.path.join(chunked_reference_base, f"{cell_type_1}.h5ad")
    cell_type_2_path = os.path.join(chunked_reference_base, f"{cell_type_2}.h5ad")

    assert os.path.exists(cell_type_1_path), f"{cell_type_1_path} DNE"
    assert os.path.exists(cell_type_2_path), f"{cell_type_2_path} DNE"

    cell_type_1_adata = ad.read_h5ad(cell_type_1_path)
    cell_type_2_adata = ad.read_h5ad(cell_type_2_path)
    
    # subset each object to the min of cell_per_type cells or the number of cells in the object
    n_cells_cell_type_1 = cell_type_1_adata.shape[0]
    n_cells_cell_type_2 = cell_type_2_adata.shape[0]


    if equal_size_n:
      min_size = np.min(n_cells_cell_type_1, n_cells_cell_type_2)
      if min_size < cell_per_type:
        n_to_sample_to = min_size
      else:
        n_to_sample_to = cell_per_type
      cell_type_1_adata =  cell_type_1_adata[np.random.choice(n_cells_cell_type_1, n_to_sample_to, replace=False), :].copy()
      cell_type_2_adata =  cell_type_2_adata[np.random.choice(n_cells_cell_type_2, n_to_sample_to, replace=False), :].copy()
    else:
      if n_cells_cell_type_1 > cell_per_type:
          cell_type_1_adata = cell_type_1_adata[np.random.choice(n_cells_cell_type_1, cell_per_type, replace=False), :].copy()
      if n_cells_cell_type_2 > cell_per_type:
          cell_type_2_adata = cell_type_2_adata[np.random.choice(n_cells_cell_type_2, cell_per_type, replace=False), :].copy()
  
    #normalize log both objects
    sc.pp.normalize_total(cell_type_1_adata, target_sum=1e4)
    sc.pp.log1p(cell_type_1_adata)

    sc.pp.normalize_total(cell_type_2_adata, target_sum=1e4)
    sc.pp.log1p(cell_type_2_adata)

    cell_type_1_mat_markers_only = cell_type_1_adata[:, markers].X.toarray()
    cell_type_2_mat_markers_only = cell_type_2_adata[:, markers].X.toarray()

    knn_classifer = KNeighborsClassifier(n_neighbors=12, metric="cosine", weights="distance")
    reference_labels = [cell_type_1] * cell_type_1_mat_markers_only.shape[0] + [cell_type_2] * cell_type_2_mat_markers_only.shape[0]
    training_data = np.vstack([cell_type_1_mat_markers_only, cell_type_2_mat_markers_only])
    knn_classifer.fit(training_data, reference_labels)
    return knn_classifer


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


def preform_pairwise_knn_prediction( model_path, n_neighbors_h0, query_obj, combined_obj = None, n_next_cell_types_compare = None, n_cores = 5, embed_key="X_pca", ref_ct_label = "ClusterNm", models_are_folder_chunked=True):

    assert query_obj is not None, "query_obj must be provided"

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

    # run knn prediction
    if n_next_cell_types_compare is None:
        # run pairwise comparisons
        print("Running pairwise comparisons for cellular predictions, n_next_cell_types_compare not provided")
        h0_predictions = preform_knn_prediction_from_combined(combined_obj, batch_var = "source", ref_value = "reference", ref_label = ref_ct_label, n_neighbors=n_neighbors_h0, embed_key = embed_key)
        cell_type_comparisons = identify_cell_type_comparisons(h0_predictions)
    else:
        print("Running n_groups comparisons for cellular predictions. n_next_cell_types_compare provided.")
        h0_predictions = None
        # print the time
        print(time.ctime())
        cell_type_comparisons = create_cellular_predictions_and_comparisons_n_groups(combined_obj, k_for_top_ct = n_neighbors_h0, n_neighbor_cts_needed = n_next_cell_types_compare, embed_key=embed_key, ref_ct_label=ref_ct_label)

    gc.collect()
    # print time
    print(time.ctime())
    knn_ensemble_results = run_knn_ensemble_comparisons_parallel(cell_type_comparisons, model_path, query_obj, n_cores = n_cores, models_are_folder_chunked=models_are_folder_chunked)
    print(time.ctime())
    print("Parsing first round results")
    gc.collect()
    cell_type_comparisons_failed = identify_failed_cell_type_comparisons(knn_ensemble_results)
    print(time.ctime())
    gc.collect()
    print("Running secondary comparisons")
    seconary_results = run_knn_ensemble_comparisons_from_cell_type_comparisons(cell_type_comparisons_failed, query_obj, model_path, n_cores = n_cores, models_are_folder_chunked=models_are_folder_chunked)
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
    ref_labels_vec = full_ref_labels_vec.iloc[cell_indices_vec]
    # order the cell types for top k by frequency
    top_k_cts = ref_labels_vec[:k_for_top_ct].value_counts()
    top_ct_names = top_k_cts.index
    # get top cell type and its p for top k
    top_ct = top_ct_names[0]
    top_ct_prop = top_k_cts.iloc[0] / k_for_top_ct

    cts_non_top = ref_labels_vec[ref_labels_vec != top_ct]
    _, first_seen_inx = np.unique(cts_non_top, return_index=True)
    sorted_first_seen_inx=np.sort(first_seen_inx)
    first_seen_non_top_cts = cts_non_top.iloc[sorted_first_seen_inx]
    first_seen_non_self_cts = first_seen_non_top_cts[:n_neighbor_cts_needed].tolist()

    res = {"prediction": top_ct, "cell_name": cell_name, "confidence": top_ct_prop, "possible_cell_types": first_seen_non_self_cts}
    return res

def run_knn_ensemble_comparisons_parallel(cell_type_comparisons, model_path, query_obj, n_cores = 30, models_are_folder_chunked=True):
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
    
def run_knn_ensemble_comparisons_from_cell_type_comparisons(cell_type_comparisons_failed,  query_obj, model_path, n_cores = 30, models_are_folder_chunked=True):
# construct arguments
    gene_label_vector = query_obj.var.index
    n_cell_type_comparisons_needed = len(cell_type_comparisons_failed)
    print(f"Constructing args for {n_cell_type_comparisons_needed} cell type comparisons")


    args = [(ele["cell_name"], ele["prediction"], ele["possible_cell_types"], pd.Series(query_obj[ele["cell_name"], :].X.toarray().flatten(), index=gene_label_vector), model_path, inx, models_are_folder_chunked) for inx, ele in enumerate(cell_type_comparisons_failed)]
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(predict_individual_cell_ensemble_models, args)
    return results

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
            knn_model_object = read_in_model(predicted_cell_type, compare_cell_type, model_path_base, suffix="classifier")
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
