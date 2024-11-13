import argparse
import os
import pandas as pd
import numpy as np
import anndata as ad
import json
import warnings

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--json_specs', type=str, help='Path to json file with parameters')
args = parser.parse_args()

with open(args.json_specs) as f:
    specs = json.load(f)



cell_type_col = specs.get('cell_type_col', None)
whole_reference_path = specs.get('whole_reference_path', None)
general_working_dir = specs.get('general_working_dir', None)
chunked_reference_out_path = specs.get("chunked_reference_out_path", None)
min_cell_type_n = specs.get("min_cell_type_n", None)
n_cells_per_chunked_obj = specs.get("n_cells_per_chunked_obj", None)
donor_col = specs.get("donor_col", None)



if cell_type_col is None:
    raise Exception("cell_type_col is None")
if whole_reference_path is None:
    raise Exception("whole_reference_path is None")
if min_cell_type_n is None:
    raise Exception("min_cell_type_n is None")
if general_working_dir is None:
    raise Exception("general_working_dir is None")
if n_cells_per_chunked_obj is None:
    raise Exception("n_cells_per_chunked_obj is None")
if chunked_reference_out_path is None:
    raise Exception("chunked_reference_out_path is None")



# load progress and check if step is already done
progress_file = os.path.join(general_working_dir, 'progress.json')
with open(progress_file) as f:
    progress = json.load(f)

current_status = progress["chunk_reference"]
if current_status:
    print("chunk_reference already done")
    exit(0)

print("valid inputs.")

# chunked_reference_out_path = os.path.join(general_working_dir, "chunked_reference")
os.makedirs(chunked_reference_out_path, exist_ok=True)
print("made chunked_reference_out_path")

print("Reading in reference")
whole_ref_obj = ad.read_h5ad(whole_reference_path)


split_by_donor=not donor_col is None
key_split_key = "_$_"
if split_by_donor:
    
    assert donor_col in whole_ref_obj.obs.columns, f"{donor_col} is not in ref"
    cell_type_col_new = "donor_cell_type"
    whole_ref_obj.obs[donor_col] =  whole_ref_obj.obs[donor_col].astype(str)
    whole_ref_obj.obs[cell_type_col] = whole_ref_obj.obs[cell_type_col].astype(str)
    whole_ref_obj.obs[cell_type_col_new] = whole_ref_obj.obs[donor_col] + key_split_key + whole_ref_obj.obs[cell_type_col]
    cell_type_col = cell_type_col_new
    
assert cell_type_col in whole_ref_obj.obs.columns, f"{cell_type_col} is not in ref"

unique_cell_types, sizes = np.unique(whole_ref_obj.obs[cell_type_col], return_counts=True)




for inx, (cell_type, cell_type_size) in enumerate(zip(unique_cell_types, sizes)):
    if inx % 10 == 0:
        print(inx)
    if cell_type_size < min_cell_type_n:
        continue

    print(cell_type)
    print(cell_type_size)

    is_ct = whole_ref_obj.obs[cell_type_col] == cell_type
    indices = np.array(np.where(is_ct)).flatten()
    n_to_sample = np.min([len(indices), n_cells_per_chunked_obj])
    sampled_indices = np.random.choice(a=indices, size=n_to_sample, replace=False)

    sampling_bool = np.zeros(whole_ref_obj.shape[0]).astype(bool)    
    sampling_bool[sampled_indices] = True

    sub_obj = whole_ref_obj[sampling_bool]
    print(sub_obj.shape)
    

    if split_by_donor:
        donor_cell_type_lis = cell_type.split(key_split_key)
        assert len(donor_cell_type_lis) == 2, f"{cell_type}"
        donor_id = donor_cell_type_lis[0]
        cell_type_id = donor_cell_type_lis[1]
        donor_chunked_path = os.path.join(chunked_reference_out_path, donor_id)
        if not os.path.isdir(donor_chunked_path):
            os.makedirs(donor_chunked_path, exist_ok=True)
        ct_out_path = os.path.join(donor_chunked_path, f"{cell_type_id}.h5ad")
    else:
        ct_out_path = os.path.join(chunked_reference_out_path, f"{cell_type}.h5ad")
    print(ct_out_path)
    sub_obj.write_h5ad(ct_out_path)



import time
time.sleep(5)

# update progress
print('Updating progress')
progress['chunk_reference'] = True
with open(progress_file, 'w') as f:
    json.dump(progress, f)


print("done.")

