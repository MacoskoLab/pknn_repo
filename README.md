# Repo housing code for running and training pKNN Models (Pairwise K Nearest Neighbor)

Note: Code here is written around being runnable on Broad's interal compute cluster.


The pKNN leverages pairwise differential expression results to make fine grained distinctions between similiar cell types.

The preparation for using this approach consists of 3 steps:
    1.) Identify pairwise markers
    2.) Fit pairwise classifiers using these markers and reference
    3.) Create subsampled reference


Code for running these steps are in the **create_models** folder and detailed in the **example_training_steps** notebook.

