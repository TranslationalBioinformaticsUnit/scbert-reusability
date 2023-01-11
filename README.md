# scBERT-reusability
This repository contains the user experience of reproducing the scBERT model for single-cell annotation (https://doi.org/10.1038/s42256-022-00534-z). It also contains the code and the online data of running this model in a new dataset and analyzing the effect of the distribution based on the number of cells per cell type.

* [Reproducibility of results](#1-reproducibility-of-results)
* [Effect of distribution](#2-effect-of-distribution)

## 1-Reproducibility of results
### 1.1-Installation
The code was downloaded from the original GitHub (https://github.com/TencentAILabHealthcare/scBERT). Then, Python 3.6.8 and the required libraries (from its [requeriments.txt](https://github.com/TencentAILabHealthcare/scBERT/blob/master/requirements.txt) file) were installed. However, you will need to install two additional Python libraries to be able to run scBERT:
```	
python -m pip install einops==0.4.1
python -m pip install local_attention==1.4.4
```	
Additionally, you should ask for access to the following relevant files:

* Pretrained model: *panglao_pretrain.pth*
* Panglao dataset for preprocessing step: *panglao_10000.h5ad*
* Preprocessed dataset: *Zheng68K.h5ad*
	
### 1.2-Training
After finishing the previous step, you can train the model using the following parameters:
```
python -m torch.distributed.launch finetune.py --data_path "Zheng68K.h5ad" --model_path "panglao_pretrain.pth"
```
By default, the code is programmed using one fold cross-validation. Increase the parameter *n_splits* (*finetune.py#lineXXX*) to use more than one. In the publication and this reproducibility, we set it to 5.
Computationally, using one NVIDIA V100 GPU it takes approximately 3 days just to finish one fold. In case to have more than one, we highly recommend executing in parallel.
### 1.3-Prediction
The best model, based on the accuracy, obtained in the training step is used for prediction. Run the following the command.
```
python predict.py --data_path "Zheng68K.h5ad" --model_path "./ckpts/finetune_best.pth"
```
In less than one hour you will take the result.
### 1.4-Detection of novel cell types
In the original publication, the dataset Macparland was used for the prediction of novel cell type. You download the data, *GSE115469_CellClusterType.txt.gz* and *GSE115469_Data.csv.gz* from [GSE115469](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115469), create *h5ad* format and preprocess it by running the script [preprocess.py](https://github.com/TencentAILabHealthcare/scBERT/blob/master/preprocess.py) of the original GitHub. We trained the model by removing the cell types denominated in the publication (*Mature_B_Cells, Plasma_Cells, alpha-beta_T_Cells, gamma-delta_T_Cells_1, gamma-delta_T_Cells_2*) and detect them as novel cell type using the following command:
```
python predict.py --data_path "test_data_path.h5ad" --model_path "finetuned_model_path" --novel_type True --unassign_thres 0.5  
```
Note: if the *test_data_path.h5ad* has different number of classes (cell types) than the training dataset, you will obtain an error (*size mismatch*) so you should adjust the parameter *out_dim* (_predict.py#lineXXX_) and set it to the number of classes of the training dataset.
## 2-Effect of distribution
### 2.1-Dataset
The different datasets can be found in this GitHub:

* Neurips_dataset: preprocessed original dataset.
* Neurips_subsampling: reducing the number of cells to 300 of all cell types.
* Neurips_oversampling: augmenting the number of cells of *BP* and *MoP* to 4800 cells using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) algorithm, function *fit_resample* and *seed=2021*.
	
### 2.2-Analysis
For this analysis, we needed to update the Python, to 3.8.1, and libraries versions to be able to use this dataset, see requirements_update.txt. With the updated version all the results of the publication can be reproduced too. We used the same parameters and command as in the reproducibility.
