# scBERT-reusability
This repository contains the user experience of reproducing the scBERT model for single-cell annotation (https://doi.org/10.1038/s42256-022-00534-z). It also contains the code and the online data of running this model in a new dataset and analyzing the effect of the distribution based on the number of cells per cell type.

1-Reproducibility of results

2-Effect of distribution

## 1-Reproducibility results
#### 1.1-Installation
The code was downloaded from the original GitHub (https://github.com/TencentAILabHealthcare/scBERT). Then, python 3.6.8 and the required libraries (from its requeriments.txt file) were installed. However, you will need to install two additional python libraries to be able to run scBERT code:
```	
python -m pip install einops==0.4.1
python -m pip install local_attention==1.4.4
```	
Additionally, you should ask for access to the following relevant files:

* pretrained-model: panglao
* prepocessed dataset: Zheng68K
* gene2gene vec
	
#### 1.2-Training
After finishing the previous step, you can train the model using the following command line:

python -m torch.distributed.launch --data_path "fine-tune_data_path" --model_path "pretrained_model_path" finetune.py

By default, the code is programmed using one fold cross-validation. Increase the variable n-split in line xxxx to use more than one. In the publication and this analysis, we set it to 5.
Computationally, using one NVIDIA V100 GPU it takes approximately 3 days just to finish one fold. In case to have more than one, we highly recommend executing in parallel.
#### 1.3-Prediciton
The best model, based on the accuracy, obtained in the training step is used for prediction. Run the following the command.
python....
In less than one hour you will take the result.
#### 1.4-Prediction novel celltype
In the original publication, the dataset Macparland was used for the prediction of novel cell type. You download the data from GSE115469, write in h5ad format and preprocess it by running the available script in the original GitHub preprocess.py. (We share the macparland dataset prepocessed??) We trained the model by removing the cell types denominated in the publication (Mature_B_Cells,Plasma_Cells,alpha-beta_T_Cells,gamma-delta_T_Cells_1,gamma-delta_T_Cells_2) and detect them as novel cell type running the following parameters:
	python
## 2-Effect of distribution
#### 2.1-Dataset
The different datasets can be found in this GitHub:

* Neurips_dataset: original dataset.
* Neurips_subsampling: reducing the number of cells to 300 of all cell types.
* Neurips_oversampling: augmented the number of cell to 4800 of BP and MoP cells using SMOTE algorithm.
	
#### 2.2-Analysis
For this analysis, we needed to update the python (from 3.6.1 to 3.8.1) and libraries versions to be able to use this dataset, see requirements_update.txt. With the updated version all the results of the publication can be reproduced too. We used the same parameters and command as in the reproducibility.
