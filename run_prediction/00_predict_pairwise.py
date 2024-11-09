

import sys
import pandas as pd
import anndata as ad
import pickle
import os
import argparse
import logging
from pathlib import Path
sys.path.append('/broad/macosko/jsilverm/pknn_repo/')
import pairwise_functions as pf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def parse_args():
    parser = argparse.ArgumentParser(description='PKNN Analysis Script')
    parser.add_argument("--query_path", type=str, required=True,
                      help="Path to query H5AD file")
    parser.add_argument("--run_id", type=str, required=True,
                      help="Unique identifier for this run")
    parser.add_argument("--reference_path", type=str, required=True,
                      help="Path to reference H5AD file")
    parser.add_argument("--out_dir", type=str, required=True,
                      help="Output directory for results")
    parser.add_argument("--ct_col", type=str, required=True,
                      help="Cell type column name in reference data")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to PKNN model")
    parser.add_argument("--n_cores", type=int, default=5,
                      help="Number of cores to use (default: 5)")
    parser.add_argument("--n_neighbors_h0", type=int, default=10,
                      help="Number of neighbors for H0 (default: 10)")
    parser.add_argument("--n_next_cell_types_compare", type=int, default=5,
                      help="Number of next cell types to compare (default: 5)")
    return parser.parse_args()

def check_paths(args):
    """Validate input paths and create output directory."""
    # Check input files exist
    for path_name, path in [("Query", args.query_path), 
                          ("Reference", args.reference_path),
                          ("Model", args.model_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} file not found: {path}")
    
    # Create output directory if needed
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check if output file already exists
    out_pkl_name = os.path.join(args.out_dir, f"{args.run_id}.pkl")
    if os.path.exists(out_pkl_name):
        raise FileExistsError(f"Output file already exists: {out_pkl_name}")
    
    return out_pkl_name

def load_anndata(query_path, reference_path, cell_type_col):
    """Load and validate AnnData objects."""
    try:
        query_obj = ad.read_h5ad(query_path)
        ref_obj = ad.read_h5ad(reference_path)
        
        # Validate cell type column
        if cell_type_col not in ref_obj.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_col}' not found in reference data")
        
        # Add source labels
        query_obj.obs["source"] = "query"
        ref_obj.obs["source"] = "reference"
        
        return query_obj, ref_obj
    
    except Exception as e:
        logging.error(f"Error loading AnnData objects: {str(e)}")
        raise

def process_shared_genes(query_obj, ref_obj):
    """Process and align shared genes between query and reference."""
    genes_ref = ref_obj.var.index
    genes_query = query_obj.var.index
    shared_genes = genes_ref.intersection(genes_query)
    
    if len(shared_genes) == 0:
        raise ValueError("No shared genes found between query and reference")
    
    ref_obj = ref_obj[:, shared_genes]
    query_obj = query_obj[:, shared_genes]
    
    logging.info(f"Query genes: {len(genes_query)}")
    logging.info(f"Reference genes: {len(genes_ref)}")
    logging.info(f"Shared genes: {len(shared_genes)}")
    
    if not all(ref_obj.var.index == query_obj.var.index):
        raise ValueError("Gene alignment error after subsetting")
    
    return query_obj, ref_obj

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Log parameters
        logging.info("Starting PKNN analysis with parameters:")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
        
        # Check paths and get output path
        out_pkl_name = check_paths(args)
        
        # Load data
        query_obj, ref_obj = load_anndata(args.query_path, args.reference_path, args.ct_col)
        logging.info(f"Loaded query: {query_obj}")
        logging.info(f"Loaded reference: {ref_obj}")
        # Process shared genes
        query_obj, ref_obj = process_shared_genes(query_obj, ref_obj)
        
        # Combine objects and run clustering
        logging.info("Combining objects and running clustering workflow")
        combined_obj = ad.concat([query_obj, ref_obj], join="outer")
        combined_obj = pf.run_clustering_workflow(combined_obj)
        
        # Run PKNN prediction
        logging.info("Running PKNN prediction")
        pairwise_pred_results = pf.preform_pairwise_knn_prediction(
            model_path=args.model_path,
            n_neighbors_h0=args.n_neighbors_h0,
            query_obj=query_obj,
            combined_obj=combined_obj,
            n_next_cell_types_compare=args.n_next_cell_types_compare,
            n_cores=args.n_cores,
            embed_key="X_pca",
            ref_ct_label=args.ct_col
        )
        
        # Save results
        logging.info(f"Saving results to {out_pkl_name}")
        with open(out_pkl_name, "wb") as f:
            pickle.dump(pairwise_pred_results, f)
        
        logging.info("Analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()