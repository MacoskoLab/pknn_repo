#!/bin/bash

#SBATCH --time=1-24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --nodes=1


# Wrapper to run all steps of model creation
podman_img_name="synapse_seq_img"
# take in path to spec file
spec_file=$1
working_dir=$2
tracking_path=$3

# remove trailing / from working_dir
working_dir=$(echo $working_dir | sed 's:/*$::')

if [ -z "$spec_file" ]
then
    echo "Usage: create_models.sh <spec_file> <working_dir> <tracking_path>"
    echo "spec_file not found"
    exit 1
fi

if [ -z "$working_dir" ]
then
    echo "Usage: create_models.sh <spec_file> <working_dir> <tracking_path>"
    echo "working_dir not found"
    exit 1
fi

if [ -z "$tracking_path" ]
then
    echo "Usage: create_models.sh <spec_file> <working_dir> <tracking_path>"
    echo "tracking_path not found"
    exit 1
fi

# Run 00_chunk_reference.py
chunking_log_file="${working_dir}/chunking.log"
cmd="podman run -v /broad/macosko:/broad/macosko ${podman_img_name} python -u /broad/macosko/jsilverm/pknn_repo/create_models/00_chunk_reference.py -i ${spec_file} > ${chunking_log_file} 2>&1"
echo $cmd
eval $cmd


# Run 01_run_pairwise_de_batch.py
differntial_expression_log_file="${working_dir}/differential_expression.log"
cmd="podman run -v /broad/macosko:/broad/macosko ${podman_img_name} python -u /broad/macosko/jsilverm/pknn_repo/create_models/01_run_pairwise_de_batch.py -i ${spec_file} > ${differntial_expression_log_file} 2>&1"
echo $cmd
eval $cmd

# Run 02_create_pairwise_de_models.py
create_models_log_file="${working_dir}/create_models.log"
cmd="podman run -v /broad/macosko:/broad/macosko ${podman_img_name} python -u /broad/macosko/jsilverm/pknn_repo/create_models/02_create_pairwise_models.py -i ${spec_file} > ${create_models_log_file} 2>&1"
echo $cmd
eval $cmd

# Run 03_create_reference_sampled.py
create_ref_log_file="${working_dir}/create_ref.log"
cmd="podman run -v /broad/macosko:/broad/macosko ${podman_img_name} python -u /broad/macosko/jsilverm/pknn_repo/create_models/03_create_reference_sampled.py -i ${spec_file} > ${create_ref_log_file} 2>&1"
echo $cmd
eval $cmd
