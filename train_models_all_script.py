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

n_cores = 220

sys.path.append('/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/07_transfer_benchmarking/')
import knn_pairwise_methods as pknn

markers_base="/home/jsilverm/MouseBrainAtlasData/IntegrationObjects/03_full_atlas_pairiwse_markers_name_fixed"
chunked_reference_base="/home/jsilverm/MouseBrainAtlasData/IntegrationObjects/00_chunked_references/raw_leaf_downsampled"
classifier_out_path = "/mnt/disks/allen-data/pknn_macosko_full_mb_first"

ct_list_path = "/home/jsilverm/MouseBrainAtlasData/IntegrationObjects/01_Allen_Atlas/Midbrain/cts_needed_for_mb.csv"
cts_df = pd.read_csv(ct_list_path)
cell_types_present = list(cts_df["0"])


for ct in cell_types_present:
    marker_path = os.path.join(markers_base, f"{ct}.pkl")
    
    exists = os.path.exists(marker_path)
    if not exists:
        print(ct)

# files_present = os.listdir(markers_base)
# cell_types_present = [ele.split(".pkl")[0] for ele in files_present]

if not os.path.exists(classifier_out_path):
    os.makedirs(classifier_out_path)

args_create_knn = [(inx, cell_type, cell_types_present[inx+1:], chunked_reference_base, markers_base, classifier_out_path, False) for inx, cell_type in enumerate(cell_types_present)]


pool = mp.Pool(n_cores)
pool.starmap(pknn.create_classifier_set, args_create_knn)
pool.close()
pool.join()
