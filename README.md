
This repository contains the code and data for our paper "[**AlphaPPIMI: A Comprehensive Deep Learning Framework for Predicting PPI-Modulator Interactions**]". Currently, the paper is under review. We plan to make the code publicly available after the paper is accepted.  ！！！


# AlphaPPIMI

**AlphaPPIMI: A Comprehensive Deep Learning Framework for Predicting PPI-Modulator Interactions**




## Model Architecture of AlphaPPIMI

<div align="center">
  <img src="image/AlphaPPIMI_framework.png" alt="AlphaPPIMI Architecture" width="800">
</div>


## Acknowledgements



## News!

under review ！！！  

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

*  Jianmin Wang, Jiashun Mao, Chunyan Li, Hongxin Xiang, Xun Wang, Shuang Wang, Zixu Wang, Yangyang Chen, Yuquan Li, Kyoung Tai No, Tao Song, Xiangxiang Zeng; Interface-aware molecular generative framework for protein-protein interaction modulators.  J Cheminform (2024). doi: https://doi.org/10.1186/s13321-024-00930-0

*  Cankara, Fatma, et al. "DiPPI: A Curated Data Set for Drug-like Molecules in Protein–Protein Interfaces." Journal of Chemical Information and Modeling 64.13 (2024): 5041-5051. https://doi.org/10.1021/acs.jcim.3c01905  






