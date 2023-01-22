# scBERT-reusability
This repository contains the user experience of reproducing the scBERT results for single-cell annotation (https://doi.org/10.1038/s42256-022-00534-z). It also contains the material for analyzing the effect of the distribution of cells per cell type. For further details, we recommend you to read the reusability report (*link*).

* [Reproducibility of results](#1-reproducibility-of-results)
* [Effect of distribution of cells](#2-effect-of-distribution-of-cells)

## 1-Reproducibility of results
### 1.1-Installation
The code was downloaded from the original GitHub (https://github.com/TencentAILabHealthcare/scBERT). Then, Python 3.6.8 and the required libraries (see [requirements.txt](https://github.com/TencentAILabHealthcare/scBERT/blob/master/requirements.txt) file) were installed. However, you will need to install two additional Python libraries to be able to run scBERT:
```	
python -m pip install einops==0.4.1
python -m pip install local_attention==1.4.4
```	
Additionally, you should ask the authors for access to the following relevant files (see [data section](https://github.com/TencentAILabHealthcare/scBERT#data) for more details):

* Pretrained model: *panglao_pretrain.pth*
* Panglao dataset for preprocessing step: *panglao_10000.h5ad*
* Preprocessed dataset: *Zheng68K.h5ad*

Then, some bugs from the original scripts needed to be solved to make the results reproducible.
* *[finetune.py](https://github.com/TencentAILabHealthcare/scBERT/blob/master/finetune.py)*: update it to run using more than one fold-validation.
* *[predict.py](https://github.com/TencentAILabHealthcare/scBERT/blob/master/predict.py)*: solve *IndexError* in *line124*.

The updated scripts are accesible in this GitHub.
### 1.2-Training
After finishing the installation step, you can train the model using the following parameters:
```
python -m torch.distributed.launch finetune_updated.py --data_path "Zheng68K.h5ad" --model_path "panglao_pretrain.pth"
```
By default, the process is programmed for using five fold cross-validation as in the publication. To modify it, you should update the parameter *n_splits* in *[finetune_updated.py](https://github.com/TranslationalBioinformaticsUnit/scbert-reusability/blob/main/finetune_updated.py)#line144*.
Computationally, using one NVIDIA V100 GPU it takes approximately 3 days just to finish one fold with *Zheng68K* dataset. For this reason, we highly recommend executing each fold in parallel for computational optimization.
### 1.3-Prediction
The best model, based on the accuracy, obtained in the training step, is used for prediction. Run the following the command:
```
python predict_updated.py --data_path "Zheng68K.h5ad" --model_path "./ckpts/finetune1_best.pth"
```
Execution time is quite fast (less than one hour).
### 1.4-Detection of novel cell types
In the publication, the dataset *Macparland* was used for the detection of novel cell type. You should download the data, *GSE115469_CellClusterType.txt.gz* and *GSE115469_Data.csv.gz* from [GSE115469](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115469), generate *h5ad* format and preprocess it by running the script [preprocess.py](https://github.com/TencentAILabHealthcare/scBERT/blob/master/preprocess.py) of the original GitHub. Later, to train the model by removing the cell types denominated in the publication (*Mature_B_Cells, Plasma_Cells, alpha-beta_T_Cells, gamma-delta_T_Cells_1, gamma-delta_T_Cells_2*) and, finally, detect them as novel cell type using the following parameters:
```
python predict_updated.py --data_path "test_data_path.h5ad" --model_path "finetuned_model_path" --novel_type True --unassign_thres 0.5  
```
*Note:* if the *test_data_path.h5ad* has different number of classes (cell types) than the training dataset, you will obtain an error (*size mismatch*) so you should adjust the parameter *out_dim* in *[predict_updated.py](https://github.com/TranslationalBioinformaticsUnit/scbert-reusability/blob/main/predict_updated.py)#line97* and set it to the number of classes of the training dataset.
## 2-Effect of distribution of cells
### 2.1-Data
The following datasets were used for studying the effect of distribution of cells and they can be downloaded [here](https://figshare.com/projects/scbert-reusability/157203):

* *Neurips_dataset*: preprocessed original dataset.
* *Neurips_subsampling*: reducing the number of cells to 300 of all cell types.
* *Neurips_oversampling*: augmenting the number of *BP* and *MoP* to 4800 cells using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) algorithm, function *fit_resample* and *seed=2021*.
	
### 2.2-Analysis
For this analysis, we needed to update the Python, to 3.8.1, and libraries versions to be able to use this dataset, see [requirements_update.txt](https://github.com/TranslationalBioinformaticsUnit/scbert-reusability/blob/main/requirements_update.txt). With these updated versions, all the publication results can be reproduced too. We used the same parameters and commands as in the [reproducibility section](#1-reproducibility-of-results), except, *[finetune.py](https://github.com/TencentAILabHealthcare/scBERT/blob/master/finetune.py)#line124* should be replaced by the following:
```
label_dict, label = np.unique(np.array(data.obs['cell_type']), return_inverse=True)
```

