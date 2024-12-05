
import multiprocessing as mp
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import pickle
import os
import gc
import time
import scipy
import sys

from sklearn.neighbors import KNeighborsClassifier


def compute_balanced_mean_markers(cell_type_1_adata, cell_type_2_adata, n_genes_dir=25):
    na_val=1
    cell_type_1_adata.layers["counts"] = cell_type_1_adata.X
    cell_type_2_adata.layers["counts"] = cell_type_2_adata.X
    
    sc.pp.normalize_total(cell_type_1_adata)
    sc.pp.normalize_total(cell_type_2_adata)
    
    sc.pp.log1p(cell_type_1_adata)
    sc.pp.log1p(cell_type_2_adata)
    
    cell_type_1_adata.layers["data"] = cell_type_1_adata.X
    cell_type_2_adata.layers["data"] = cell_type_2_adata.X
    
    
    cell_type_1_markers = []
    cell_type_2_markers = []
    
    weighting_cell_type_1 = np.ones(cell_type_1_adata.shape[0])
    weighting_cell_type_2 = np.ones(cell_type_2_adata.shape[0])
    
    
    current_adata_1 = cell_type_1_adata.copy()
    current_adata_2 = cell_type_2_adata.copy()
    
    for gene_inx in range(n_genes_dir):
        assert np.all(current_adata_1.var_names == current_adata_2.var_names)
        gene_names = current_adata_1.var_names
    
        cell_type_1_weighted_mat = (current_adata_1.X.tocsr()).multiply(weighting_cell_type_1.reshape(-1, 1))
        cell_type_2_weighted_mat = (current_adata_2.X.tocsr()).multiply(weighting_cell_type_2.reshape(-1, 1))
        
        cell_type_1_nonzero_series = pd.Series(np.array(np.mean(cell_type_1_weighted_mat, axis=0)).flatten(), index=gene_names)
        cell_type_2_nonzero_series = pd.Series(np.array(np.mean(cell_type_2_weighted_mat, axis=0)).flatten(), index=gene_names)
        
        group_1_pos_diff = cell_type_1_nonzero_series - cell_type_2_nonzero_series
        # top gene is top positive marker for cell_type_1
        # bottom gene is top positive marker for cell_type_2
        group_1_pos_sorted = group_1_pos_diff.sort_values(ascending=False)
    
        # take top and bottom markers
        cell_type_1_marker_name = group_1_pos_sorted.index[0]
        cell_type_1_marker_diff = group_1_pos_sorted.iloc[0]
        
        cell_type_2_marker_name = group_1_pos_sorted.index[group_1_pos_sorted.shape[0]-1]
        cell_type_2_marker_diff = group_1_pos_sorted.iloc[group_1_pos_sorted.shape[0]-1]
    
        cell_type_1_markers.append(cell_type_1_marker_name)
        cell_type_2_markers.append(cell_type_2_marker_name)
    
        gene_order = np.array(cell_type_1_markers + cell_type_2_markers)
        
        cell_type_1_observed_values = cell_type_1_adata[:,gene_order].X.toarray()
        reference_profile_ct_1 = np.array(list(np.ones(len(cell_type_1_markers))) + list(np.zeros(len(cell_type_2_markers)))).reshape(1, -1)
        weighting_cell_type_1 = scipy.spatial.distance.cdist(cell_type_1_observed_values, reference_profile_ct_1, metric="cosine").flatten()
        weighting_cell_type_1[np.isnan(weighting_cell_type_1)] = na_val
    
        cell_type_2_observed_values = cell_type_2_adata[:,gene_order].X.toarray()
        reference_profile_ct_2 = np.array(list(np.zeros(len(cell_type_1_markers))) + list(np.ones(len(cell_type_2_markers)))).reshape(1, -1)
        weighting_cell_type_2 = scipy.spatial.distance.cdist(cell_type_2_observed_values, reference_profile_ct_2, metric="cosine").flatten()
        weighting_cell_type_2[np.isnan(weighting_cell_type_2)] = na_val
    
        # remove genes from both objects
        selected_genes = set(gene_order)
    
        is_valid_gene_ct_1 = ~current_adata_1.var_names.isin(selected_genes)
        is_valid_gene_ct_2 = ~current_adata_2.var_names.isin(selected_genes)
        
        current_adata_1 = current_adata_1[:,is_valid_gene_ct_1]
        current_adata_2 = current_adata_2[:,is_valid_gene_ct_2]
    
    group_1_markers = pd.Series(np.arange(len(cell_type_1_markers)), index=cell_type_1_markers)
    group_2_markers = pd.Series(np.arange(len(cell_type_2_markers)), index=cell_type_2_markers)
    return group_1_markers, group_2_markers

def compute_gene_rankings_nonzero(cell_type_1_adata, cell_type_2_adata):
    assert np.all(cell_type_1_adata.var_names == cell_type_2_adata.var_names)
    gene_names = cell_type_1_adata.var_names
    cell_type_1_values_series = pd.Series(np.array(np.mean(cell_type_1_adata.X > 0, axis=0)).flatten(), index=gene_names)
    cell_type_2_values_series = pd.Series(np.array(np.mean(cell_type_2_adata.X > 0, axis=0)).flatten(), index=gene_names)

    cell_type_1_diff_2 = cell_type_1_values_series - cell_type_2_values_series
    cell_type_1_diff_2_sorted = pd.DataFrame(cell_type_1_diff_2.sort_values(ascending=False), columns=["score"])
    cell_type_1_diff_2_sorted["gene"] = cell_type_1_diff_2_sorted.index
    cell_type_1_diff_2_sorted["ranking"] = np.arange(0, cell_type_1_diff_2_sorted.shape[0])
    cell_type_1_diff_2_sorted = cell_type_1_diff_2_sorted.reset_index(drop=True)

    cell_type_2_diff_1 = cell_type_2_values_series - cell_type_1_values_series
    cell_type_2_diff_1_sorted = pd.DataFrame(cell_type_2_diff_1.sort_values(ascending=False), columns=["score"])
    cell_type_2_diff_1_sorted["gene"] = cell_type_2_diff_1.index
    cell_type_2_diff_1_sorted["ranking"] = np.arange(0, cell_type_2_diff_1_sorted.shape[0])
    cell_type_2_diff_1_sorted = cell_type_2_diff_1_sorted.reset_index(drop=True)

    res = {"cell_type_1": cell_type_1_diff_2_sorted, "cell_type_2": cell_type_2_diff_1_sorted}
    
    return res



def compute_and_save_markers_donor_chunked(base_chunked_dir, cell_type_1, cell_type_2, out_dir_base, marker_comp_method="nonzero", valid_markers_set=None, donor_consensus_method="mean"):
    assert marker_comp_method in ["nonzero"], f"{marker_comp_method} not valid"
    assert donor_consensus_method in ["max", "mean"], f"{donor_consensus_method} not valid"

    print("identifying valid donors")
    # Read in all donor split cell type objects
    donor_dirs_present = os.listdir(base_chunked_dir)
    
    possible_cell_type_1_paths = [os.path.join(base_chunked_dir, donor_dir, f"{cell_type_1}.h5ad") for donor_dir in donor_dirs_present]
    cell_type_1_paths = [ct_1_path for ct_1_path in possible_cell_type_1_paths if os.path.exists(ct_1_path)]
    assert len(cell_type_1_paths) != 0, f"No paths found for {cell_type_1}"
    
    possible_cell_type_2_paths = [os.path.join(base_chunked_dir, donor_dir, f"{cell_type_2}.h5ad") for donor_dir in donor_dirs_present]
    cell_type_2_paths = [ct_2_path for ct_2_path in possible_cell_type_2_paths if os.path.exists(ct_2_path)]
    assert len(cell_type_2_paths) != 0, f"No paths found for {cell_type_2}"
    
    print("reading in objects")
    cell_type_1_objs = {}
    for cell_type_1_path in cell_type_1_paths:
        donor = cell_type_1_path.split("/")[-2]
        obj = ad.read_h5ad(cell_type_1_path)

        if valid_markers_set is not None:
            is_valid_gene = obj.var_names.isin(valid_markers_set)
            obj = obj[:,is_valid_gene]
        cell_type_1_objs[donor] = obj.copy()
    
    cell_type_2_objs = {}
    for cell_type_2_path in cell_type_2_paths:
        donor = cell_type_2_path.split("/")[-2]
        obj = ad.read_h5ad(cell_type_2_path)
        if valid_markers_set is not None:
            is_valid_gene = obj.var_names.isin(valid_markers_set)
            obj = obj[:,is_valid_gene]
        cell_type_2_objs[donor] = obj.copy()

    print("computing rankings")
    ct_1_rankings = []
    ct_2_rankings = []
    
    for ct1_inx, (cell_type_1_donor, ct_1_obj) in enumerate(cell_type_1_objs.items()):
        for ct2_inx, (cell_type_2_donor, ct_2_obj) in enumerate(cell_type_2_objs.items()):
            rankings = compute_gene_rankings_nonzero(ct_1_obj, ct_2_obj)
            ct_1_current_df = rankings["cell_type_1"]
            ct_1_current_df["cell_type_1_donor"] = cell_type_1_donor
            ct_1_current_df["cell_type_2_donor"] = cell_type_2_donor
            ct_1_rankings.append(ct_1_current_df)
            
            ct_2_current_df = rankings["cell_type_2"]
            ct_2_current_df["cell_type_1_donor"] = cell_type_1_donor
            ct_2_current_df["cell_type_2_donor"] = cell_type_2_donor
            ct_2_rankings.append(ct_2_current_df)
    
    ct_1_concat = pd.concat(ct_1_rankings).reset_index(drop=True)
    ct_2_concat = pd.concat(ct_2_rankings).reset_index(drop=True)

    print("aggregating rankings")
    # hardcoded savings of 1k genes
    if donor_consensus_method == "max":
        ct_1_rankings = ct_1_concat.groupby("gene")["ranking"].max().sort_values().iloc[:1000]
        ct_2_rankings = ct_2_concat.groupby("gene")["ranking"].max().sort_values().iloc[:1000]
    elif donor_consensus_method == "mean":
        ct_1_rankings = ct_1_concat.groupby("gene")["ranking"].mean().sort_values().iloc[:1000]
        ct_2_rankings = ct_2_concat.groupby("gene")["ranking"].mean().sort_values().iloc[:1000]
    
    res = {cell_type_1: ct_1_rankings, cell_type_2: ct_2_rankings}
    
    save_pairwise_model(cell_type_1, cell_type_2, out_dir_base, res, "markers")
    gc.collect()



def compute_and_save_markers(base_chunked_dir, cell_type_1, cell_type_2, n_genes, out_dir_base, marker_comp_method="nonzero", valid_markers_set=None):
    valid_marker_methods = ["nonzero", "mean", "balanced_mean"]
    if marker_comp_method not in valid_marker_methods:
        raise Exception(f"marker_comp_method: {marker_comp_method} is not in valid methods: {valid_marker_methods}")
        
    # load objs
    cell_type_1_path = os.path.join(base_chunked_dir, f"{cell_type_1}.h5ad")
    cell_type_2_path = os.path.join(base_chunked_dir, f"{cell_type_2}.h5ad")

    assert os.path.exists(cell_type_1_path), f"{cell_type_1_path} DNE"
    assert os.path.exists(cell_type_2_path), f"{cell_type_2_path} DNE"

    cell_type_1_adata = ad.read_h5ad(cell_type_1_path)
    cell_type_2_adata = ad.read_h5ad(cell_type_2_path)
    

    assert np.all(cell_type_1_adata.var_names == cell_type_2_adata.var_names)
    gene_names = cell_type_1_adata.var_names
    if valid_markers_set is not None:
        is_valid_feature = cell_type_1_adata.var_names.isin(valid_markers_set)
        cell_type_1_adata = cell_type_1_adata[:,is_valid_feature]
        cell_type_2_adata = cell_type_2_adata[:,is_valid_feature]


    if marker_comp_method == "balanced_mean":
        group_1_markers, group_2_markers = compute_balanced_mean_markers(cell_type_1_adata, cell_type_2_adata, n_genes_dir=n_genes)
    else:
        if marker_comp_method == "nonzero":
            cell_type_1_values_series = pd.Series(np.array(np.mean(cell_type_1_adata.X > 0, axis=0)).flatten(), index=gene_names)
            cell_type_2_values_series = pd.Series(np.array(np.mean(cell_type_2_adata.X > 0, axis=0)).flatten(), index=gene_names)
        elif marker_comp_method == "mean":
            cell_type_1_values_series = pd.Series(np.array(np.mean(cell_type_1_adata.X, axis=0)).flatten(), index=gene_names)
            cell_type_2_values_series = pd.Series(np.array(np.mean(cell_type_2_adata.X, axis=0)).flatten(), index=gene_names)
        
        group_1_pos_diff = cell_type_1_values_series - cell_type_2_values_series
        group_1_pos_sorted = group_1_pos_diff.sort_values(ascending=False)
        group_1_markers = group_1_pos_sorted[:n_genes]
        
        group_2_pos_diff = cell_type_2_values_series - cell_type_1_values_series
        group_2_pos_sorted = group_2_pos_diff.sort_values(ascending=False)
        group_2_markers = group_2_pos_sorted[:n_genes]
    
    res = {cell_type_1: group_1_markers, cell_type_2: group_2_markers}

    save_pairwise_model(cell_type_1, cell_type_2, out_dir_base, res, "markers")


def compute_markers_cell_type_to_all(chunked_ref_base, cell_type_1, cell_type_2_lis, n_markers_dir, markers_out_path_full, marker_comp_method, valid_markers_set=None, is_donor_chunked=False):
    valid_marker_methods = ["nonzero", "mean", "balanced_mean"]
    if marker_comp_method not in valid_marker_methods:
        raise Exception(f"marker_comp_method: {marker_comp_method} is not in valid methods: {valid_marker_methods}")
    
    for cell_type_2 in cell_type_2_lis:
        if is_donor_chunked:
            print("using donor chunked")
            compute_and_save_markers_donor_chunked(chunked_ref_base, cell_type_1, cell_type_2, markers_out_path_full, marker_comp_method, valid_markers_set, "max")
        else:    
            compute_and_save_markers(chunked_ref_base, cell_type_1, cell_type_2, n_markers_dir, markers_out_path_full, marker_comp_method, valid_markers_set)
    

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

def get_markers_chunked(cell_type_1, cell_type_2, markers_base, suffix=None, n_markers_dir=25):

    obj = read_in_sorted_subfolder_obj(cell_type_1, cell_type_2, markers_base, suffix=suffix)
    markers_1 = obj[cell_type_1][:n_markers_dir]
    markers_2 = obj[cell_type_2][:n_markers_dir]
    
    combined_markers_index = markers_1.index.append(markers_2.index)
    # Convert to NumPy array
    combined_markers = np.array(combined_markers_index)
    return combined_markers


def create_classifier_set(inx, current_cell_type, compare_cell_types, chunked_reference_base, markers_base, base_classifier_path, n_features_directional=25, cell_per_type=200, equal_size_n=False, classifier_name="knn"):
    print(current_cell_type)
    f_name="classifier"
    if inx % 10 == 0:
        print(f"Working on inx: {inx}")
    if not os.path.exists(base_classifier_path):
        os.makedirs(base_classifier_path, exist_ok=True)
    for cell_type in compare_cell_types:
        model_exists = check_if_model_exists(current_cell_type, cell_type, base_classifier_path, f_name)
        if model_exists:
            continue
        try:
            create_paired_classifer_test_n_markers(current_cell_type, cell_type, chunked_reference_base, markers_base, base_classifier_path, cell_per_type=cell_per_type, classifier_name=classifier_name)
        except Exception as e:
            print("ERROR in create paired classifier set!")
            print(f"cell type 1: {current_cell_type}")
            print(f"cell type 2: {cell_type}")
            print(e)
            sys.exit(1)
        
def create_classifier_set_non_donor_split(inx, current_cell_type, compare_cell_types, chunked_reference_base, markers_base, base_classifier_path, n_features_directional=25, cell_per_type=200, equal_size_n=False):
    
    f_name="classifier"
    if inx % 10 == 0:
        print(f"Working on inx: {inx}")
    if not os.path.exists(base_classifier_path):
        os.makedirs(base_classifier_path, exist_ok=True)
    for cell_type in compare_cell_types:
        model_exists = check_if_model_exists(current_cell_type, cell_type, base_classifier_path,f_name)
        if model_exists:
            continue

        markers = get_markers_chunked(current_cell_type, cell_type, markers_base, suffix="markers", n_markers_dir=n_features_directional)

        classifier = create_classifier_pair(cell_type_1 = current_cell_type, cell_type_2=cell_type, chunked_reference_base=chunked_reference_base, 
                                            markers=markers, cell_per_type=cell_per_type, 
                                            equal_size_n=equal_size_n)
        res_obj = {"classifier": classifier, "cell_type1": cell_type, "cell_type2": current_cell_type,  "markers": markers}

        save_pairwise_model(current_cell_type, cell_type, base_classifier_path, res_obj, f_name, order_alphabetically=True)




def read_in_donor_chunked_objs(base_chunked_dir, cell_type, valid_markers_set=None, normalize=False):
    donor_dirs_present = os.listdir(base_chunked_dir)
    possible_cell_type_paths = [os.path.join(base_chunked_dir, donor_dir, f"{cell_type}.h5ad") for donor_dir in donor_dirs_present]
    cell_type_paths = [ct_path for ct_path in possible_cell_type_paths if os.path.exists(ct_path)]
    
    assert len(cell_type_paths) != 0, f"No paths found for {cell_type}"

    print("reading in objects")
    cell_type_objs = {}
    for cell_type_path in cell_type_paths:
        donor = cell_type_path.split("/")[-2]
        obj = ad.read_h5ad(cell_type_path)

        if valid_markers_set is not None:
            is_valid_gene = obj.var_names.isin(valid_markers_set)
            obj = obj[:,is_valid_gene]
        if normalize:
            obj.layers["counts"] = obj.X
            sc.pp.normalize_total(obj, target_sum=1e4)
            sc.pp.log1p(obj)
            obj.layers["data"] = obj.X
            
        cell_type_objs[donor] = obj
    return cell_type_objs



def create_donor_balanced_ref(cell_type_objs, target_size=300, seed=6):
    np.random.seed(seed)
    n_donors_total = len(cell_type_objs)

    size_per_donor = [(cell_type_objs[donor_id].shape[0], donor_id) for donor_id in cell_type_objs.keys()]

    size_per_donor = sorted(size_per_donor)
    n_sample_per_donor = {}
    n_remaining_to_sample = target_size
    
    for donor_inx, (n_per_donor, donor_id) in enumerate(size_per_donor):
        n_donors_remain = n_donors_total - donor_inx
        target_donor_size = int(np.ceil(n_remaining_to_sample / n_donors_remain))
        if n_per_donor <= target_donor_size:
            n_sample_per_donor[donor_id] = n_per_donor
            n_remaining_to_sample -= n_per_donor
        else:
            n_sample_per_donor[donor_id] = target_donor_size            
            n_remaining_to_sample -= target_donor_size

    sampled_adata_objs = {}
    for donor_id, n_to_sample in n_sample_per_donor.items():
        donor_adata = cell_type_objs[donor_id]
        sampled_indices = np.random.choice(donor_adata.shape[0], n_to_sample, replace=False)
        sampled_adata_objs[donor_id] = donor_adata[sampled_indices]

    combined_sampled_obj = ad.concat(sampled_adata_objs, axis=0)
    
    return combined_sampled_obj


def create_pred_model(cell_type_1, cell_type_2, cell_type_1_adata, cell_type_2_adata, markers, classifier_name="svm"):

    valid_classifiers = ["knn", "svm"]
    assert classifier_name in valid_classifiers, f"classifier_name must be in {valid_classifiers}"

    cell_type_1_mat_markers_only = cell_type_1_adata[:, markers].X.toarray()
    cell_type_2_mat_markers_only = cell_type_2_adata[:, markers].X.toarray()

    if classifier_name == "svm":
        classifier = SVC(
            kernel='rbf',  # Radial basis function kernel
            probability=True,  # Enable probability estimates
            random_state=42,
            class_weight='balanced'
        )
    elif classifier_name == "knn":
        classifier = KNeighborsClassifier(n_neighbors=12, metric="cosine", weights="distance")
        
    # Prepare training labels
    reference_labels = ([cell_type_1] * cell_type_1_mat_markers_only.shape[0] + 
                       [cell_type_2] * cell_type_2_mat_markers_only.shape[0])
    
    # Combine training data
    training_data = np.vstack([cell_type_1_mat_markers_only, cell_type_2_mat_markers_only])
    
    # Fit the classifier
    classifier.fit(training_data, reference_labels)
    
    res_obj = {"classifier": classifier, "cell_type1": cell_type_1, "cell_type2": cell_type_2,  "markers": markers}
    return res_obj

def predict_obj_from_classifier_obj(query_obj, query_cell_type, classifier_obj):
    """Assumes query_obj is normalized"""
    markers = classifier_obj["markers"]
    model = classifier_obj["classifier"]

    query_data_mat = query_obj[:,markers].X.toarray()

    class_labels = model.classes_
    if len(class_labels) != 2:
        raise ValueError("This function expects a binary classifier")
         
    # Get predictions and probabilities
    predictions = model.predict(query_data_mat)
    prediction_probs = model.predict_proba(query_data_mat)
    
    # Get probability for the positive class (confidence)
    
    # Calculate accuracy
    correct_mask = predictions == query_cell_type
    accuracy = np.mean(correct_mask)
    
    positive_class_idx = np.where(class_labels == query_cell_type)[0][0]
    negative_class_idx = np.where(class_labels != query_cell_type)[0][0]
    mean_correct_minus_incorrect_p = np.mean(prediction_probs[:,positive_class_idx] -  prediction_probs[:,negative_class_idx])

    res ={"query_cell_type":query_cell_type, "accuracy": accuracy, "conf_metric": mean_correct_minus_incorrect_p}

    
    return res


def create_paired_classifer_test_n_markers(cell_type_1, cell_type_2, chunked_reference_base, markers_base, base_classifier_path, cell_per_type=300, min_markers=5, max_markers=150, n_jump_markers=5, classifier_name="knn"):
    
    print(f"Read in {cell_type_1} objs")
    cell_type_1_objs = read_in_donor_chunked_objs(chunked_reference_base, cell_type_1, normalize=True)
    print(f"Read in {cell_type_2} objs")
    cell_type_2_objs = read_in_donor_chunked_objs(chunked_reference_base, cell_type_2, normalize=True)
    
    marker_objs =  read_in_sorted_subfolder_obj(cell_type_1, cell_type_2, markers_base, suffix="markers")
    
    donors_ct_1 = list(cell_type_1_objs.keys())
    donors_ct_2 = list(cell_type_2_objs.keys())
    
    ct_1_heldout_dict = {"donor_witheld": [], "accuracy": [], "conf_metric": [], "n_genes_ct_1": [], "n_genes_ct_2": []}
    ct_2_heldout_dict = {"donor_witheld": [], "accuracy": [], "conf_metric": [], "n_genes_ct_1": [], "n_genes_ct_2": []}
    
    for n_genes_dir in np.arange(min_markers, max_markers, n_jump_markers):
        print(n_genes_dir)
        n_genes_ct1=n_genes_dir
        n_genes_ct2=n_genes_dir
    
        cell_type_1_markers = marker_objs[cell_type_1].sort_values(ascending=True).iloc[:n_genes_ct1]
        cell_type_2_markers = marker_objs[cell_type_2].sort_values(ascending=True).iloc[:n_genes_ct2]
        combined_markers_index = cell_type_1_markers.index.append(cell_type_2_markers.index)
        combined_markers = np.array(combined_markers_index)
        
        for inx, ct_1_donor in enumerate(donors_ct_1):
            query_obj = cell_type_1_objs.pop(ct_1_donor)
            ct_1_reference = create_donor_balanced_ref(cell_type_1_objs, cell_per_type)
            ct_2_reference = create_donor_balanced_ref(cell_type_2_objs, cell_per_type)
        
            current_classifier_obj = create_pred_model(cell_type_1, cell_type_2, ct_1_reference, ct_2_reference, combined_markers, classifier_name=classifier_name)
            predictions = predict_obj_from_classifier_obj(query_obj, cell_type_1, current_classifier_obj)
        
            ct_1_heldout_dict["donor_witheld"].append(ct_1_donor)
            ct_1_heldout_dict["accuracy"].append(predictions["accuracy"])
            ct_1_heldout_dict["conf_metric"].append(predictions["conf_metric"])
            ct_1_heldout_dict["n_genes_ct_1"].append(n_genes_ct1)
            ct_1_heldout_dict["n_genes_ct_2"].append(n_genes_ct2)
        
            cell_type_1_objs[ct_1_donor] = query_obj
        
        for inx, ct_2_donor in enumerate(donors_ct_2):
            query_obj = cell_type_2_objs.pop(ct_2_donor)
            ct_1_reference = create_donor_balanced_ref(cell_type_1_objs, cell_per_type)
            ct_2_reference = create_donor_balanced_ref(cell_type_2_objs, cell_per_type)
        
            current_classifier_obj = create_pred_model(cell_type_1, cell_type_2, ct_1_reference, ct_2_reference, combined_markers, classifier_name=classifier_name)
            predictions = predict_obj_from_classifier_obj(query_obj, cell_type_2, current_classifier_obj)
        
            ct_2_heldout_dict["donor_witheld"].append(ct_2_donor)
            ct_2_heldout_dict["accuracy"].append(predictions["accuracy"])
            ct_2_heldout_dict["conf_metric"].append(predictions["conf_metric"])
            ct_2_heldout_dict["n_genes_ct_1"].append(n_genes_ct1)
            ct_2_heldout_dict["n_genes_ct_2"].append(n_genes_ct2)
        
            cell_type_2_objs[ct_2_donor] = query_obj
    
    ct_1_heldout_df = pd.DataFrame(ct_1_heldout_dict)
    ct_2_heldout_df = pd.DataFrame(ct_2_heldout_dict)
    
    ct_1_preformance = ct_1_heldout_df.groupby("n_genes_ct_1")["conf_metric"].mean()
    ct_2_preformance = ct_2_heldout_df.groupby("n_genes_ct_1")["conf_metric"].mean()
    top_score_n_genes = (ct_1_preformance + ct_2_preformance).idxmax()
    
    acc_at_gene_selection_ct_1 = ct_1_heldout_df.loc[ct_1_heldout_df["n_genes_ct_1"] == top_score_n_genes, "accuracy"].mean()
    acc_at_gene_selection_ct_2 = ct_2_heldout_df.loc[ct_2_heldout_df["n_genes_ct_1"] == top_score_n_genes, "accuracy"].mean()
    
    
    # create final model based on results
    cell_type_1_markers = marker_objs[cell_type_1].sort_values(ascending=True).iloc[:top_score_n_genes]
    cell_type_2_markers = marker_objs[cell_type_2].sort_values(ascending=True).iloc[:top_score_n_genes]
    combined_markers_index = cell_type_1_markers.index.append(cell_type_2_markers.index)
    combined_markers = np.array(combined_markers_index)
    ct_1_reference = create_donor_balanced_ref(cell_type_1_objs, cell_per_type)
    ct_2_reference = create_donor_balanced_ref(cell_type_2_objs, cell_per_type)
    final_classifier_obj = create_pred_model(cell_type_1, cell_type_2, ct_1_reference, ct_2_reference, combined_markers, classifier_name=classifier_name)
    
    save_pairwise_model(cell_type_1, cell_type_2, base_classifier_path, final_classifier_obj, "classifier", order_alphabetically=True)

    meta_obj = {"features_selected": top_score_n_genes, "accuracy_ct_1": acc_at_gene_selection_ct_1, "accuracy_ct_2": acc_at_gene_selection_ct_2, 'n_gene_dist': top_score_n_genes}
    save_pairwise_model(cell_type_1, cell_type_2, base_classifier_path, meta_obj, "meta", order_alphabetically=True)


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
        os.makedirs(outer_folder, exist_ok=True)

    inner_folder = os.path.join(outer_folder, group2)
    if not os.path.exists(inner_folder):
        os.makedirs(inner_folder, exist_ok=True)
    
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




