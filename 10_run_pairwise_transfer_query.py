import sys

import pandas as pd
import anndata as ad
import pickle
import os
import argparse
import knn_pairwise_methods as kpm
import logging

# read in query_path reference_path out_dir n_cores n_neighbors_h0 n_next_cell_types_compare
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query_path", type=str)
parser.add_argument("-r", "--reference_path", type=str)
parser.add_argument("-o", "--out_dir", type=str)
parser.add_argument("-n", "--n_cores", type=int, default=50)
parser.add_argument("-k", "--n_neighbors_h0", type=int, default=10)
parser.add_argument("-c", "--n_next_cell_types_compare", type=int, default=None)
parser.add_argument("-i", "--index_from_end", type=int, default=1)
parser.add_argument("-s", "--from_human", action="store_true", help="If true, will use human models. Default is False.")
parser.add_argument("-m", "--ct_col", default="ClusterNm", help="")
parser.add_argument("-p", "--model_path", default="/mnt/disks/allen-data/pknn_macosko_classifiers_v1")
parser.add_argument("-f", "--models_are_folder_chunked", action="store_true", help="Bool whether models are stored in one file or in folders")


args = parser.parse_args()

query_path = args.query_path
reference_path = args.reference_path
out_dir = args.out_dir
n_cores = args.n_cores
n_neighbors_h0 = args.n_neighbors_h0
n_next_cell_types_compare = args.n_next_cell_types_compare
index_from_end = args.index_from_end
from_human = args.from_human
models_are_folder_chunked = args.models_are_folder_chunked
cell_type_col = args.ct_col
model_path = args.model_path

### Hardcoded for now! ###
query_gene_col = "gene.name"


# take last part of query_path as name
query_name = query_path.split("/")[-index_from_end].split(".h5ad")[0]
out_pkl_name = f"{out_dir}/{query_name}.pkl"

# if out_pkl_name exists, exit
if os.path.exists(out_pkl_name):
    print(f"Out pkl name {out_pkl_name} already exists. Exiting")
    sys.exit()

print(f"Query Path: {query_path}")
print(f"Query Name: {query_name}")
print(f"Out Pkl Name: {out_pkl_name}")
print(f"Reference Path: {reference_path}")
print(f"Out Dir: {out_dir}")
print(f"N Cores: {n_cores}")
print(f"N Neighbors H0: {n_neighbors_h0}")
print(f"N Next Cell Types Compare: {n_next_cell_types_compare}")
print(f"From Human: {from_human}")
print(f"Folder Chunked Model: {models_are_folder_chunked}")
print(f"cell_type_coll: {cell_type_col}")
print(f"Model Path: {model_path}")

sys.stdout.flush()
sys.stderr.flush()

# check if out_dir exists
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


assert os.path.exists(model_path)

# read in query and reference anndata objects
query_obj = ad.read_h5ad(query_path)
ref_obj = ad.read_h5ad(reference_path)

print(f"Query: {query_obj}")
print(f"Reference: {ref_obj}")

# ensure reference ct col is in columns
assert cell_type_col in ref_obj.obs.columns

if from_human:
    human_to_mouse_path = "/home/jsilverm/MouseBrainAtlasData/00_siletti_human_data/human_name_to_mouse.csv"
    human_to_mouse = pd.read_csv(human_to_mouse_path, index_col=0)
    human_to_mouse.index = human_to_mouse["human.names"]

    query_gene_name_in_df=query_obj.var[query_gene_col].isin(human_to_mouse["human.names"])
    query_obj = query_obj[:, query_gene_name_in_df]
    query_obj.var["mouse_gene"] = human_to_mouse.loc[query_obj.var[query_gene_col], "mouse.ensemble"].values
    # #ensure mouse_gene col is not na
    assert query_obj.var["mouse_gene"].isna().sum() == 0
    
    query_obj.var.index= query_obj.var["mouse_gene"]
    
    ref_obj.X = ref_obj.X.tocsc()
    
    ref_var_in_query = ref_obj.var.index.isin(query_obj.var.index)
    ref_obj = ref_obj[:, ref_var_in_query]

     # convert counts to csc matrix
    query_obj.X = query_obj.X.tocsc()
    query_var_in_ref = query_obj.var.index.isin(ref_obj.var.index)
    query_obj = query_obj[:, query_var_in_ref]
    
    # reorder query object columns to match reference
    query_obj = query_obj[:, ref_obj.var.index]
    
    assert (query_obj.var.index == ref_obj.var.index).all()
    print("asserted genes align")
  
    query_obj_csr = query_obj.X.tocsr()
    ref_obj_csr = ref_obj.X.tocsr()

    query_obj.layers["csr_copy"] = query_obj_csr
    query_obj.X = query_obj.layers["csr_copy"]

    ref_obj.layers["csr_copy"] = ref_obj_csr
    ref_obj.X = ref_obj.layers["csr_copy"]


# create combined anndata object
query_obj.obs["source"] = "query"
ref_obj.obs["source"] = "reference"

#determine if all cell types in reference have models created for them
# ref_chunked_model_training_path = "/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/07_transfer_benchmarking/objects/ref_whole_brain_chunked"
# ref_chunked_model_training_path = "/home/jsilverm/MouseBrainAtlasData/IntegrationObjects/00_chunked_references/raw_leaf_downsampled"
# ref_chunked_model_training = os.listdir(ref_chunked_model_training_path)
# cell_type_names= [x.split(".h5ad")[0] for x in ref_chunked_model_training]
# unique_types_in_ref = ref_obj.obs["ClusterNm"].unique()
# missing_types = [ele for ele in unique_types_in_ref if ele not in cell_type_names]
# is_cell_of_missing_type = ref_obj.obs["ClusterNm"].isin(missing_types)
# ref_obj = ref_obj[~is_cell_of_missing_type]

sys.stdout.flush()
sys.stderr.flush()

# subset both objects to only shared genes
genes_ref = ref_obj.var.index
genes_query = query_obj.var.index

# intersect of genes
shared_genes = genes_ref.intersection(genes_query)
ref_obj = ref_obj[:, shared_genes]
query_obj = query_obj[:, shared_genes]

print(f"n shared genes: {len(shared_genes)}")

#assert that genes match
assert all(ref_obj.var.index == query_obj.var.index), "Genes do not match"

### Combine Objects and Run Clustering Workflow to create intial embedding###
print("Combining objects and running clustering workflow")

sys.stdout.flush()
sys.stderr.flush()
combined_obj = ad.concat([query_obj, ref_obj], join="outer")
combined_obj = kpm.run_clustering_workflow(combined_obj)

sys.stdout.flush()
sys.stderr.flush()


print("Running pknn")
pairwise_pred_results = kpm.preform_pairwise_knn_prediction(model_path=model_path, n_neighbors_h0 = n_neighbors_h0, query_obj = query_obj, combined_obj = combined_obj,n_next_cell_types_compare=n_next_cell_types_compare, n_cores = n_cores, embed_key="X_pca", models_are_folder_chunked=models_are_folder_chunked, ref_ct_label=cell_type_col)

sys.stdout.flush()
sys.stderr.flush()
# pickle results to out_dir
with open(out_pkl_name, "wb") as f:
    pickle.dump(pairwise_pred_results, f)
    print(f"Saved results to {out_pkl_name}")
