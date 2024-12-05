# pKNN Model Training and Application

This repository contains code for running and training pKNN (Pairwise K-Nearest Neighbor) models. This is a python based cell type label transfer approach.

**Note:** This code is specifically designed to be executed on the Broad Institute's internal compute cluster.

## Overview

The pKNN approach uses pairwise differential expression results to enable fine-grained distinctions between similar cell types. This method is particularly effective in contexts where traditional classification approaches struggle to resolve subtle cellular differences.

## Workflow

Using pKNN involves three main steps:

1. **Identify Pairwise Markers**  
   Determine differential expression markers between each pair of cell types in the dataset.

2. **Fit Pairwise Classifiers**  
   Train classifiers using the identified markers and reference data to distinguish between each pair of cell types.

3. **Create Subsampled Reference**  
   Generate a subsampled reference dataset tailored to improve performance and computational efficiency.

## Code Structure

- The **`create_models`** folder contains scripts for executing all three steps of the workflow.
- Detailed usage examples and step-by-step instructions for training pKNN models are provided in the **`example_training_steps`** Jupyter notebook.
- The **run_prediction** folder contains files which use the trained models to predict for a query object.

## Getting Started

Refer to the **`example_training_steps`** notebook to see how to prepare input data, train models, and generate the subsampled reference for your specific use case.
