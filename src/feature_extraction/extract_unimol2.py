import pandas as pd
import numpy as np
from tqdm import tqdm
from unimol_tools import UniMolRepr
from rdkit import Chem

clf = UniMolRepr(
    data_type='molecule',
    remove_hs=False,
    model_name='unimolv2',  #  unimolv1, unimolv2
    model_size='84m',       #  unimolv2: 84m, 164m, 310m, 570m, 1.1B
)

file_path = "compound_phy.tsv"  
data = pd.read_csv(file_path, sep="\t")
smiles_list = data['SMILES'].tolist()

valid_smiles = []
for smi in tqdm(smiles_list, desc="Validating SMILES"):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        valid_smiles.append(smi)
    else:
        print(f"Invalid SMILES skipped: {smi}")

print(f"Total valid SMILES: {len(valid_smiles)}")

output_file = "smiles_features_fixed.csv"  

print("Processing all valid SMILES...")
all_features = []

fixed_cls_dim = 768  
fixed_atomic_dim = 128  

try:
    unimol_repr = clf.get_repr(valid_smiles, return_atomic_reprs=True)

    for i, smiles in enumerate(tqdm(valid_smiles, desc="Storing Features")):
        try:
            cls_repr = unimol_repr['cls_repr'][i]
            atomic_reprs = unimol_repr['atomic_reprs'][i]

            if len(cls_repr) > fixed_cls_dim:
                cls_repr = cls_repr[:fixed_cls_dim]
            else:
                cls_repr = np.pad(cls_repr, (0, fixed_cls_dim - len(cls_repr)), constant_values=0)

            if len(atomic_reprs) > fixed_atomic_dim:
                atomic_reprs = atomic_reprs[:fixed_atomic_dim]
            else:
                atomic_reprs = np.pad(atomic_reprs, ((0, fixed_atomic_dim - len(atomic_reprs)), (0, 0)), constant_values=0)

            all_features.append({
                'SMILES': smiles,
                'cls_repr': ','.join(map(str, cls_repr)),  
                'atomic_reprs': ';'.join(
                    [','.join(map(str, atom)) for atom in atomic_reprs]
                )  
            })
        except Exception as e:
            print(f"Error storing features for SMILES: {smiles}, Error: {e}")
except Exception as e:
    print(f"Error during feature extraction: {e}")

features_df = pd.DataFrame(all_features)
features_df.to_csv(output_file, index=False)
print(f"All features saved to {output_file}")






