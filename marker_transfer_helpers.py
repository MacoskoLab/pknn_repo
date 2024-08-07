import pandas as pd
import anndata as ad
import os

def convert_ensemble_to_gene_name(ensemble_id):
    var_df = pd.read_csv("/home/jsilverm/MouseBrainAtlasData/IntegrationFunctions/00_IntegrationPipe/TestMarkerFunctions/chunked_references/ensemble_gene_name_conversion.csv", index_col=0)
    gene_name = var_df.loc[ensemble_id, "gene_name"]
    return(gene_name)


def read_clade_obj(clade_name, clade_df = None, chunked_obj_base = None):
    """
    Using a chunked version of the mouse atlas, where each object in a directory corresponds to a leaf node in the clade tree, read in all objects that are in a given clade
    clade_name: str name of the clade to read in
    clade_df: pandas dataframe of the clade tree. If None, will read in from clade_df_path
    clade_df_path: str path to the clade_df
    chunked_obj_base: str path to the directory where the chunked objects are stored
    """
    
    if clade_df is None:
        clade_df = load_clade_df()
        # clade_df = pd.read_csv(clade_df_path, index_col=1)

    if chunked_obj_base is None:
        raise Exception("Must provide chunked_obj_base path")

    is_clade = clade_df["clade_name"] == clade_name
    leaf_names = clade_df[is_clade].index.values
    obj_list = []
    for leaf_name in leaf_names:
        obj_path = chunked_obj_base + "/" + leaf_name + ".h5ad"
        #if the file exists read in it, otherwise skip and output name
        try:
            obj = ad.read_h5ad(obj_path)
            #combine current index and derived_cell_libs
            # obj.obs["cell_name_unique"] = obj.obs.index + obj.obs["derived_cell_libs"]
            obj_list.append(obj)
        except:
            continue
            # print("Could not find " + obj_path)
    
    if len(obj_list) == 0:
        return None

    clade_obj = combine_obj_atlas(obj_list)
    return clade_obj

def read_leaf_obj(leaf_name, chunked_obj_base, from_macosko = True):
    """
    Read in a single object from the chunked objects
    leaf_name: str name of the leaf to read in
    chunked_obj_base: str path to the directory where the chunked objects are stored
    """
    # convert - to _ in leaf name
    if from_macosko:
        leaf_name = leaf_name.replace("-", "_")
    obj_path = os.path.join(chunked_obj_base, leaf_name + ".h5ad")
    #ensure object exists
    if not os.path.exists(obj_path):
        raise Exception(f"Could not find {obj_path}")
    obj = ad.read_h5ad(obj_path)
    return obj


def combine_obj_atlas(anndata_list):
    """
    Combine a list of anndata objects into a single object. Will not make indices unique.
    anndata_list: list of anndata objects
    """
    if len(anndata_list) == 0:
        raise Exception("Must provide at least one anndata object")
    if len(anndata_list) == 1:
        return anndata_list[0]
    combined_obj = ad.concat(anndata_list, join="outer", index_unique=None)
    return combined_obj


def load_clade_df(clade_df_path = "~/MouseBrainAtlasData/IntegrationObjects/IntegrationPipeObjects/clade_df_ng_fixed.csv"):
    """
    Load in the clade dataframe
    clade_df_path: str path to the clade_df
    """
    clade_df = pd.read_csv(clade_df_path, index_col=1)
    return clade_df

