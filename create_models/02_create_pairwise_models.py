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


def get_all_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get full path
            
            # full_path = os.path.join(root, file)
            files_list.append(file)
    return files_list

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--json_specs', type=str, help='Path to json file with parameters')
args = parser.parse_args()

with open(args.json_specs) as f:
    specs = json.load(f)


chunked_reference_out_path = specs.get('chunked_reference_out_path', None)
markers_reference_out_path = specs.get('markers_reference_out_path', None)
classifier_out_path = specs.get('classifier_out_path', None)
general_working_dir = specs.get('general_working_dir', None)
n_genes_directional_models = specs.get('n_genes_directional_models', None)
n_cores = specs.get('n_cores', None)
max_cells_per_ct_model = specs.get('max_cells_per_ct_model', None)
classifier_name = specs.get('classifier_name', None)


assert chunked_reference_out_path is not None, 'chunked_reference_out_path is required'
assert markers_reference_out_path is not None, 'markers_reference_out_path is required'
assert general_working_dir is not None, 'general_working_dir is required'
# assert n_genes_directional_models is not None, 'n_genes_directional_models is required'
assert n_cores is not None, 'n_cores is required'
assert max_cells_per_ct_model is not None, 'max_cells_per_ct_model is required'

progress_file = os.path.join(general_working_dir, 'progress.json')
with open(progress_file) as f:
    progress = json.load(f)


chunked_reference_done = progress['chunk_reference']
if not chunked_reference_done:
    print('chunk_reference not done yet')
    exit(0)

marker_computation = progress['marker_computation']
if not marker_computation:
    print('marker_computation not done yet')
    exit(0)

status = progress['model_creation']
if status:
    print('model_creation already done')
    exit(0)



equal_size_n=False
is_donor_chunked=True

if is_donor_chunked:
    all_terminal_files = get_all_files(chunked_reference_out_path)
    cell_types_present = list(set([fname.split(".h5ad")[0] for fname in all_terminal_files if ".h5ad" in fname]))
else:
    chunked_objs_present = os.listdir(chunked_reference_out_path)
    cell_types_present = [fname.split(".h5ad")[0] for fname in chunked_objs_present if ".h5ad" in fname]


print(cell_types_present)


args_create_models = [(inx, cell_type, cell_types_present[inx+1:], chunked_reference_out_path,
                    markers_reference_out_path, classifier_out_path, n_genes_directional_models,
                     max_cells_per_ct_model, equal_size_n, classifier_name) for inx, cell_type in enumerate(cell_types_present)]


pool = mp.Pool(n_cores)
pool.starmap(pf.create_classifier_set, args_create_models)
pool.close()
pool.join()


print('Finished creating models')
progress['model_creation'] = True
with open(progress_file, 'w') as f:
    json.dump(progress, f)

print('Done!')
