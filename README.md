[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/Bigrock-dd/AlphaPPIMI)
[![ J. Cheminform.](https://img.shields.io/badge/%20J.%20Cheminform.%20(2025)-red)](https://doi.org/10.1186/s13321-025-01077-2)


This repository contains the code and data for our paper "**AlphaPPIMI: A Comprehensive Deep Learning Framework for Predicting PPI-Modulator Interactions**".


# AlphaPPIMI

**AlphaPPIMI: A Comprehensive Deep Learning Framework for Predicting PPI-Modulator Interactions**

Protein-protein interactions (PPIs) regulate essential biological processes through complex interfaces, with their dysfunction is associated with various
diseases. Consequently, the identification of PPIs and their interface-targeting modulators has emerged as a critical therapeutic approach. However, discovering
modulators that target PPIs and PPI interfaces remains challenging as traditional structure-similarity-based methods fail to effectively characterize
PPI targets, particularly those for which no active compounds are known. Here, we present AlphaPPIMI, a comprehensive deep learning framework that
combines large-scale pretrained language models with domain adaptation for predicting PPI-modulator interactions, specifically targeting PPI interface. To
enable robust model development and evaluation, we constructed comprehensive benchmark datasets of PPI-modulator interactions (PPIMI). Our framework
integrates comprehensive molecular features from Uni-Mol2, protein representations derived from state-of-the-art language models (ESM2 and ProTrans), and
PPI structural characteristics encoded by PFeature. Through a specialized crossattention architecture and conditional domain adversarial networks (CDAN), AlphaPPIMI effectively learns potential associations between PPI targets and modulators while ensuring robust cross-domain generalization. Extensive evaluations demonstrate that AlphaPPIMI significantly outperforms existing methods
in predicting PPIMI, providing a powerful tool for identifying novel PPI modulators, particularly those acting on PPI interfaces.


## Model Architecture of AlphaPPIMI

<div align="center">
  <img src="image/AlphaPPIMI_framework.png" alt="AlphaPPIMI Architecture" width="800">
</div>


## Acknowledgements



## News!

**[2025/08/08]** Accepted in **Journal of Cheminformatics**, 2025.  

**[2025/03/08]** submission to **Journal of Cheminformatics**, 2025.  


## Requirements
```
#Basic dependencies
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
torch-geometric>=2.0.0
scikit-learn>=0.24.0
tqdm>=4.62.0
#Molecular processing
rdkit>=2022.03.1
#Deep learning models
transformers>=4.12.0
fair-esm>=2.0.0
protbert>=0.1.0
```

## Data



The original datasets can be found or downloaded from the following sources:

- [DiPPI: Drugs in Protein-Protein Interfaces](http://interactome.ku.edu.tr:8501/)  
- [DLiP: Database of Chemical Library for Protein-Protein Interaction](https://skb-insilico.com/dlip)  
- [iPPI-DB: Database of Modulators of Protein-Protein Interactions](https://ippidb.pasteur.fr/)

Note: Processed data will be made available upon paper acceptance.






## Usage

The code will be made available upon paper acceptance. The framework supports two training modes:

### Additional Data for Domain Adaptation

To support domain adaptation experiments, a feature file is required. Please follow the steps below:

 1. Download the file from the following Google Drive link:
 
      [Download Feature File](https://drive.google.com/file/d/1ImF9DnvqUS8RFt8SllbRYF10XPxrrnwU/view?usp=drive_link)

 2. Place the downloaded file into the following directories:
 
	•	data/domain_adaptaion/source/features
 
	•	data/domain_adaptaion/target/DiPPI/features

Make sure the file is present in both directories so that the domain adaptation mode can correctly load the required features.

### Standard Training Mode
In standard training mode, we evaluate the model using two data split strategies:

1. Random Split (random pairs of compounds and proteins):
```
python main.py --fold 1 --eval_setting random --batch_size 64  --epochs 200
```
2. Cold-pair Split (unseen compounds and proteins):
```
python main.py --fold 1 --eval_setting cold --batch_size 64  --epochs 200
```

### Domain Adaptation Mode
For domain adaptation, the data is divided into source and target domains. You can train the model with different target datasets:

For example, using DiPPI as target domain:
```
python main.py --fold 1  --use_domain_adaptation --target_dataset DiPPI
```
Key arguments:
- `--fold`: Fold number for cross validation
- `--eval_setting`: Data split strategy [random/cold]
- `--target_dataset`: Target domain selection [DiPPI/iPPIDB]
- `--use_domain_adaptation`: Use domain_adaptation for training


## License
Code is released under MIT LICENSE.


## Cite:

*   Dayan Liu, Tao Song, Shuang Wang, Xue Li, Peifu Han, Jianmin Wang*, Shudong Wang*; AlphaPPIMI: A Comprehensive Deep Learning Framework for Predicting PPI-Modulator Interactions. J Cheminform 17, 134 (2025). doi: https://doi.org/10.1186/s13321-025-01077-2  
*  Jianmin Wang, Jiashun Mao, Chunyan Li, Hongxin Xiang, Xun Wang, Shuang Wang, Zixu Wang, Yangyang Chen, Yuquan Li, Kyoung Tai No*, Tao Song*, Xiangxiang Zeng*; Interface-aware molecular generative framework for protein-protein interaction modulators.  J Cheminform (2024). doi: https://doi.org/10.1186/s13321-024-00930-0

*  Cankara, Fatma, et al. "DiPPI: A Curated Data Set for Drug-like Molecules in Protein–Protein Interfaces." Journal of Chemical Information and Modeling 64.13 (2024): 5041-5051. https://doi.org/10.1021/acs.jcim.3c01905  






