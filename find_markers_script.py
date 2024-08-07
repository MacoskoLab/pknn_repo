import sys
sys.path.append("/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/06_marker_based_transfer/02_marker_based_transfer_methods")

import matplotlib.pyplot as plt
import multiprocessing as mp
import anndata as ad
import pandas as pd
import scanpy as sc
import argparse
import scipy.stats as stats
import marker_transfer_helpers as th
import marker_computation_functions as mcf
import comparison_functions as cf
import numpy as np
import pickle
import sys
import os
import seaborn as sns

sys.path.append('/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/07_transfer_benchmarking/')
import knn_pairwise_methods as pknn



pct_nonzero_df_path = "/home/jsilverm/MouseBrainAtlasData/IntegrationObjects/MouseBrainMarkerObjects/leaf_pct_nonzero_5030_nodes.csv"
pct_nonzero_df = pd.read_csv(pct_nonzero_df_path, index_col=0)
pct_nonzero_df = pct_nonzero_df.drop("group_name", axis=1)

def compute_and_save_pct_diff_markers(cell_type_name, all_cell_type_names, pct_nonzero_df, out_path, n_genes=50):
    out_path = os.path.join(out_path, cell_type_name + ".pkl")
    if os.path.exists(out_path):
        print("Already computed", cell_type_name)
        return
    marker_obj = {}
    for compare_cell_type_name in all_cell_type_names:
        if compare_cell_type_name == cell_type_name:
            continue
        diff_vals = mcf.get_max_difference_nonzero_genes_vector(cell_type_name, compare_cell_type_name, pct_nonzero_df = pct_nonzero_df, n_genes=n_genes)
        diff_df = pd.DataFrame(diff_vals, columns=["pct_diff"])
        diff_df.index.name = "ensembl_gene_id"
        marker_obj[compare_cell_type_name] = diff_df

    with open(out_path, "wb") as f:
        pickle.dump(marker_obj, f)


out_path = "/home/jsilverm/MouseBrainAtlasData/IntegrationObjects/03_full_atlas_pairiwse_markers"

all_cell_types = list(pct_nonzero_df.index)

os.makedirs(out_path, exist_ok=True)
n_cores = 125
args = [(cell_type_name, all_cell_types, pct_nonzero_df, out_path, 50) for cell_type_name in all_cell_types]
pool = mp.Pool(n_cores)
pool.starmap(compute_and_save_pct_diff_markers, args)
pool.close()
pool.join()