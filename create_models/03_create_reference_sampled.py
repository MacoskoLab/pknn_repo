import argparse
import os
import pandas as pd
import numpy as np
import anndata as ad
import multiprocessing as mp
import json
import sys
sys.path.append('/broad/macosko/jsilverm/pknn_repo/')
import pairwise_functions as pf

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--json_specs', type=str, help='Path to json file with parameters')
args = parser.parse_args()

with open(args.json_specs) as f:
    specs = json.load(f)

chunked_reference_out_path = specs.get('chunked_reference_out_path', None)
general_working_dir = specs.get('general_working_dir', None)
max_reference_cells_in_comb = specs.get("max_reference_cells_in_comb", None)
max_query_cells_in_comb = specs.get("max_query_cells_in_comb", None)
whole_reference_path = specs.get("whole_reference_path", None)
cell_type_col = specs.get("cell_type_col", None)


assert general_working_dir is not None, "general_working_dir is not defined"
assert max_reference_cells_in_comb is not None, "max_reference_cells_in_comb is not defined"
assert max_query_cells_in_comb is not None, "max_query_cells_in_comb is not defined"
assert whole_reference_path is not None, "whole_reference_path is not defined"
assert cell_type_col is not None, "cell_type_col is not defined"
assert chunked_reference_out_path is not None, "chunked_reference_out_path is not defined"

status_file = os.path.join(general_working_dir, "progress.json")
with open(status_file) as f:
    progress = json.load(f)



chunked_reference_done = progress['chunk_reference']
if not chunked_reference_done:
    print('chunk_reference not done yet')
    exit(0)

marker_computation = progress['marker_computation']
if not marker_computation:
    print('marker_computation not done yet')
    exit(0)

if not progress['model_creation']:
    print('model creation not done yet')
    exit(0)


current_status = progress["create_sampled_ref"]
if current_status:
    print("create_sampled_ref already done")
    exit(0)

# chunked_reference_out_path = os.path.join(general_working_dir, "chunked_reference")


# Read in the reference data
ref_obj = ad.read_h5ad(whole_reference_path)


cell_type_names_objs = os.listdir(chunked_reference_out_path)
cell_type_names = [x.split(".h5ad")[0] for x in cell_type_names_objs]
cell_type_sizes_dict = ref_obj.obs[cell_type_col].value_counts().to_dict()
sizes = [cell_type_sizes_dict[x] for x in cell_type_names]


n_cell_types = len(cell_type_names)

ref_indices = []
query_indices = []

types_passing = []
types_too_small = []
for inx, (cell_type, size) in enumerate(zip(cell_type_names, sizes)):
    # track progress
    if inx % 10 == 0:
        print(inx)
         
    types_passing.append(cell_type)


    size_for_ref = min([max_reference_cells_in_comb, size])
    is_cell_type_bool = ref_obj.obs[cell_type_col] == cell_type
    indices_for_cell_type = np.where(is_cell_type_bool)[0]

    # take a random sample of the indices
    indices_for_ref = np.random.choice(indices_for_cell_type, size_for_ref, replace=False)
    ref_indices.extend(list(indices_for_ref))
    is_cell_type_bool.iloc[indices_for_ref] = False

    excess_cells = size - size_for_ref

    if excess_cells > 10:
        size_for_query = min(excess_cells, max_query_cells_in_comb)
        
        indices_possible_query_sample = np.where(is_cell_type_bool)[0]
        indices_for_query = np.random.choice(indices_possible_query_sample, size_for_query, replace=False)
    
        query_indices.extend(list(indices_for_query))

query_obj = ref_obj[query_indices]
ref_obj_sampled = ref_obj[ref_indices]

assert len(set(query_obj.obs_names).intersection(ref_obj_sampled.obs_names)) == 0

reference_path = os.path.join(general_working_dir, "reference_sampled.h5ad")
query_path = os.path.join(general_working_dir, "test_query_sampled.h5ad")

# Save the reference and query objects
ref_obj_sampled.write(reference_path)
query_obj.write(query_path)

print("updating progress")
progress["create_sampled_ref"] = True
with open(status_file, 'w') as f:
    json.dump(progress, f)


print("Done")
