import argparse
import os
import pandas as pd
import numpy as np
import anndata as ad
import multiprocessing as mp
import json
import sys
import pickle
sys.path.append('/broad/macosko/jsilverm/pknn_repo/')
import pairwise_functions as pf


# Load data
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--json_specs', type=str, help='Path to json file with parameters')
args = parser.parse_args()

with open(args.json_specs) as f:
    specs = json.load(f)


chunked_reference_out_path = specs.get('chunked_reference_out_path', None)
markers_reference_out_path = specs.get('markers_reference_out_path', None)
general_working_dir = specs.get('general_working_dir', None)
n_genes_directional_compute = specs.get('n_genes_directional_compute', None)
n_cores = specs.get('n_cores', None)
marker_comp_method = specs.get('marker_comp_method', None)

assert chunked_reference_out_path is not None, 'chunked_reference_out_path is required'
assert markers_reference_out_path is not None, 'markers_reference_out_path is required'
assert general_working_dir is not None, 'general_working_dir is required'
assert n_genes_directional_compute is not None, 'n_genes_directional_compute is required'
assert n_cores is not None, 'n_cores is required'
assert marker_comp_method is not None, 'marker_comp_method is required'

# load progress and check if step is already done
progress_file = os.path.join(general_working_dir, 'progress.json')
with open(progress_file) as f:
    progress = json.load(f)


# ensure that the previous step is done
chunked_reference_done = progress['chunk_reference']
if not chunked_reference_done:
    print('chunk_reference not done yet')
    exit(0)

marker_computation = progress['marker_computation']
if marker_computation:
    print('marker_computation already done')
    exit(0)





os.makedirs(markers_reference_out_path, exist_ok=True)

chunked_objs_present = os.listdir(chunked_reference_out_path)
cell_types_present = [fname.split(".h5ad")[0] for fname in chunked_objs_present]

#### For himba benchmarking ####
valid_markers_path = "/broad/macosko/jsilverm/pknn_cell_type_preds/shared_features.pkl"
valid_markers_set = pickle.load(open(valid_markers_path, "rb"))

de_args = []
for inx_1, ct_name_1 in enumerate(cell_types_present):
    cell_type_2_lis = cell_types_present[inx_1 + 1:]
    arg = (chunked_reference_out_path, ct_name_1, cell_type_2_lis, n_genes_directional_compute, markers_reference_out_path, marker_comp_method, valid_markers_set)
    de_args.append(arg)

pool = mp.Pool(n_cores)
pool.starmap(pf.compute_markers_cell_type_to_all, de_args)
pool.close()
pool.join()

# update progress
print('Updating progress')
progress['marker_computation'] = True
with open(progress_file, 'w') as f:
    json.dump(progress, f)


print('Done!')
