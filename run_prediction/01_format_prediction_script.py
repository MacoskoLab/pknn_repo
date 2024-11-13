import argparse
import json
import os
import sys
import pandas as pd

# take in sample_sheet_path and out_dir
parser = argparse.ArgumentParser()
parser.add_argument("--samplesheet", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()


exec_path = "/broad/macosko/jsilverm/pknn_repo/run_prediction/00_predict_pairwise.py"
image_name="synapse_seq_img"
mount="/broad/macosko"


sample_sheet_path = args.samplesheet
sample_sheet_df = pd.read_csv(sample_sheet_path)

out_dir = args.output

# make dir of out_file
os.makedirs(out_dir, exist_ok=True)

cmds=[]
# iterate over rows of sample_sheet_df
for inx, row in sample_sheet_df.iterrows():

    run_id = row['run_id']
    out_path = os.path.abspath(os.path.join(out_dir, f"{run_id}.sh"))

    # get the row as a dictionary
    row_dict = row.to_dict()
    args = f"""--query_path {row_dict['query_path']}\
    --run_id {row_dict['run_id']}\
    --reference_path {row_dict['reference_path']}\
    --out_dir {row_dict['out_dir']}\
    --ct_col {row_dict['ct_col']}\
    --model_path {row_dict['model_path']}\
    --n_cores {row_dict['n_cores']}\
    --n_neighbors_h0 {row_dict['n_neighbors_h0']}\
    --n_next_cell_types_compare {row_dict['n_next_cell_types_compare']}\
    """

    # place this 
    create_cmd = f"""podman run -v {mount}:{mount} {image_name} python {exec_path} {args}"""
    # write this to out_path
    with open(out_path, 'w') as f:
        # write /bin/bash to the file
        f.write("#!/bin/bash\n")
        f.write(create_cmd)

    memory = row_dict['memory_request']
    cores = row_dict['cores_request']
    print(f"memory: {memory}, cores: {cores}")
    abs_out_path = os.path.abspath(out_path)

    full_cmd = f"""sbatch --mem={memory} --cpus-per-task={cores} --time=1-0 {out_path}"""
    cmds.append(full_cmd)

final_out_file = os.path.join(out_dir, "run_all.sh")
with open(final_out_file, 'w') as f:
    f.write("\n".join(cmds))
    

