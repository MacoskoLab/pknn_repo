
import argparse
import os
import pandas as pd
import numpy as np
import anndata as ad
import json

# Create nonzero mat
def create_nonzero_from_counts(whole_obj, cell_type, cell_type_col):
    is_cell_type = whole_obj.obs[cell_type_col] == cell_type
    cell_type_counts = np.array(np.mean(whole_obj.X[is_cell_type] > 0, axis=0)).flatten()
    return cell_type_counts

# Load data
parser = argparse.ArgumentParser()
parser.add_argument('--json_specs', type=str, help='Path to json file with parameters')
args = parser.parse_args()

with open(args.json_specs) as f:
    specs = json.load(f)

cell_type_col = specs.get('cell_type_col', None)
whole_reference_path = specs.get('whole_reference_path', None)
n_genes_directional = specs.get('n_genes_directional', None)

# if any None, raise error
if cell_type_col is None or whole_reference_path is None or n_genes_directional is None:
    # print each one
    print('cell_type_col:', cell_type_col)
    print('whole_reference_path:', whole_reference_path)
    print('n_genes_directional:', n_genes_directional)

    raise ValueError('One of the required parameters is None')


unique_cell_types = whole_reference_path.obs[cell_type_col].unique()

gene_names = np.array(whole_reference_path.var_names).flatten()

result_lis = []
cell_type_lis = []

for inx, cell_type in enumerate(unique_cell_types):
    if inx % 10 == 0:
        print(inx)
    nonzero_res = create_nonzero_from_counts(whole_reference_path, cell_type, cell_type_col)
    result_lis.append(nonzero_res)
    cell_type_lis.append(cell_type)

nonzero_df = pd.DataFrame(result_lis,index= cell_type_lis, columns=gene_names)



