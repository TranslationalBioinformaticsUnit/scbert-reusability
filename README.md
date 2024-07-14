# scBERT-reusability
[![DOI](https://zenodo.org/badge/587636837.svg)](https://zenodo.org/badge/latestdoi/587636837)

This repository contains the code and data of the reusability report of scBERT, single-cell annotation (https://doi.org/10.1038/s42256-022-00534-z). Especially, the material for analyzing the effect of the distribution of cells per cell type. For further details, we recommend you to read the reusability report [https://doi.org/10.1038/s42256-023-00757-8](https://www.nature.com/articles/s42256-023-00757-8).

* [Installation](#1-installation)
* [Format](#2-format)
* [Data](#3-data)
* [Analysis](#4-analysis)

## 1-Installation

Here are the steps to follow for a proper installation:

1- Download the code from the original GitHub (https://github.com/TencentAILabHealthcare/scBERT).

2- Clone this repository and keep the scripts on the same path as originals.
```	
git clone https://github.com/TranslationalBioinformaticsUnit/scbert-reusability.git
```	
3- Install Python 3.8.1, tested version, and the required libraries from [requirements_update.txt](https://github.com/TranslationalBioinformaticsUnit/scbert-reusability/blob/main/requirements_update.txt) file.

Additionally, you should ask the authors for access to the following relevant files (see [data section](https://github.com/TencentAILabHealthcare/scBERT#data) for more details):

* Pretrained model: *panglao_pretrain.pth*
* Panglao dataset for preprocessing step: *panglao_10000.h5ad*
* Preprocessed dataset: *Zheng68K.h5ad*
* Gene embedding: *gene2vec_16906.npy* and stored in path *../data/*

## 2-Format
* Input: single-cell RNA sequencing (scRNA-seq) data in *H5AD* format, where *variables* stored the genes symbols and *observations* the truth cell type annotations. This file should be preprocessed before training the model by running the following script [preprocess.py](https://github.com/TencentAILabHealthcare/scBERT/blob/master/preprocess.py) of the original GitHub.
* Output:
  * Training: the model with the highest accuracy in the validation dataset. The output is in *.pth* format, a popular deep learning framework using Python.
  * Prediction: the predicted labels.

## 3-Data
The following preprocessed examples were used for studying the effect of the distribution of cells and they can be downloaded [here](https://figshare.com/projects/scbert-reusability/157203):

* *Neurips_dataset*: preprocessed original dataset.
* *Neurips_subsampling*: reducing the number of cells to 300 of all cell types.
* *Neurips_oversampling*: augmenting the number of *BP* and *MoP* to 4600 cells using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) algorithm, function *fit_resample* and *seed=2021*.
* *Neurips_randomoversampling*: augmenting the number of *BP* and *MoP* to 4000 cells using [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) algorithm, function *fit_resample* and *seed=2021*.

## 4-Analysis

These NeurIPS datasets were divided into training (size 70%) and test (size 30%) data using the function *[StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)* and *seed=2021*. For benchmarking Zheng dataset, it was also divided into training (size 80%) and test (size 20%).

### 4.1-Training

After finishing the installation and downloading the data, the model was trained using fivefold cross-validation:
```
python -m torch.distributed.launch finetune_updated.py --data_path "neurips_train.h5ad" --model_path "panglao_pretrain.pth"
```
Computationally, using one NVIDIA V100 GPU it takes approximately 8 hours just to finish one fold with the original *NeurIPS* dataset. For this reason, we highly recommend executing each fold in parallel for computational optimization.

In case of training with focal loss, this is the command:
```
python -m torch.distributed.launch finetune_focalLoss.py --data_path "neurips_train.h5ad" --model_path "panglao_pretrain.pth"
```
It is a specialized loss function designed for addressing class imbalance in classification tasks, achieved by adding alpha and gamma parameters to the standard cross-entropy loss.

### 4.2-Prediction
The best model, based on the accuracy, obtained in the training step, is used for prediction. Run the following the command:
```
python predict_updated.py --data_path "neurips_test.h5ad" --model_path "./ckpts/finetune1_best.pth"
```
Execution time is quite fast (less than one hour).

### 4.3-Detection of novel cell types
The process was performed by removing one cell type in the training process and then, including it in the test data. This process was iterated for each cell type.
```
python predict_updated.py --data_path "test_data_path.h5ad" --model_path "finetuned_model_path" --novel_type True --unassign_thres 0.5  
```
*Note:* if the *test_data_path.h5ad* has different number of classes (cell types) than the training dataset, you will obtain an error (*size mismatch*) so you should adjust the parameter *out_dim* in *[predict_updated.py](https://github.com/TranslationalBioinformaticsUnit/scbert-reusability/blob/main/predict_updated.py)#line97* and set it to the number of classes of the training dataset.

