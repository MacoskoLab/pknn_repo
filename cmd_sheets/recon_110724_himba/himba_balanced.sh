#!/bin/bash
podman run -v /broad/macosko:/broad/macosko synapse_seq_img python /broad/macosko/jsilverm/pknn_repo/run_prediction/00_predict_pairwise.py --query_path /broad/macosko/jsilverm/06_query_objects/recon_11724/recon_11724.h5ad    --run_id himba_balanced    --reference_path /broad/macosko/jsilverm/pknn_cell_type_preds/protein_coding_balanced_himba/reference_sampled.h5ad    --out_dir /broad/macosko/jsilverm/pknn_cell_type_preds/protein_coding_balanced_himba/results/    --ct_col group_label_no_sep    --model_path /broad/macosko/jsilverm/pknn_cell_type_preds/HIMBA/classifiers_balanced_group_label    --n_cores 15    --n_neighbors_h0 10    --n_next_cell_types_compare 10    