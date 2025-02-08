import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.metrics import (roc_auc_score, auc, precision_recall_curve,
                           precision_score, recall_score, f1_score,
                           confusion_matrix, accuracy_score, matthews_corrcoef)
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

# Define sequence vocabulary
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i+1) for i,v in enumerate(seq_voc)}
max_seq_len = 1000

def seq_cat(prot):
    """Convert protein sequence to numerical array"""
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)  # Use 0 for unknown characters
    return x

def get_best_threshold(output, labels):
    """Get best threshold based on F1 score"""
    preds = output[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    best_threshold = thresholds[f1_scores.argmax()]
    return best_threshold

def performance_evaluation(output, labels):
    """Calculate various performance metrics"""
    output = torch.softmax(torch.from_numpy(output), dim=1)
    pred_scores = output[:, 1]
    roc_auc = roc_auc_score(labels, pred_scores)
    prec, reca, _ = precision_recall_curve(labels, pred_scores)
    aupr = auc(reca, prec)

    best_threshold = get_best_threshold(output, labels)
    pred_labels = output[:, 1] > best_threshold
    precision = precision_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    (tn, fp, fn, tp) = confusion_matrix(labels, pred_labels).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, pred_labels)

    return roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels

class ModulatorPPIDataset(Dataset):
    """
    A custom Dataset that returns:
      (fingerprint, protein_feats, cls_repr, atomic_reprs, label)

    Returns:
      - fingerprint: [nBits] (ECFP fingerprint)
      - protein_feats: [2 * protein_dim] (concatenated features of protein1 and protein2)
      - cls_repr: [768] or [1,768] (need to squeeze)
      - atomic_reprs: [num_atoms, 768] (atom-level, variable length)
      - label: int (-1 for unlabeled mode)
    """
    def __init__(self, mode, setting, fold, domain='source', use_domain_adaptation=False, data_path='./data/domain_adaptation/source/'):
        super().__init__()
        self.mode = mode
        self.setting = setting
        self.fold = fold
        self.use_domain_adaptation = use_domain_adaptation

        # Check domain parameter only when using domain adaptation
        if use_domain_adaptation:
            self.domain = domain.lower()
            if self.domain not in ['source', 'target']:
                raise ValueError("domain must be 'source' or 'target'")

            self.data_path = data_path

            if self.domain == 'target':
                # Validate target domain data paths
                required_files = [
                    os.path.join(self.data_path, 'features', 'compound_phy.tsv'),
                    os.path.join(self.data_path, 'features', 'protein_esm2.csv'),
                    os.path.join(self.data_path, 'features', 'protein_phy.csv'),
                    os.path.join(self.data_path, 'features', 'protein_protrans.csv'),
                    os.path.join(self.data_path, 'train_unlabeled.csv' if not self.is_labeled else 'test.csv')
                ]

                for file_path in required_files:
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(
                            f"Required target domain file not found: {file_path}\n"
                            f"Please ensure all data files are in place."
                        )
        else:
            self.data_path = data_path

        # Check if labeled
        if self.mode == 'train_unlabeled':
            self.is_labeled = False
        else:
            self.is_labeled = True

        # Process data
        self.process_molecule()
        self.process_protein()
        self.load_and_filter_fold_data()

    def load_and_filter_fold_data(self):
        """Load and filter data based on fold settings"""
        if self.use_domain_adaptation and self.domain == 'target':
            # Target domain data loading
            if self.is_labeled:
                datapath = os.path.join(self.data_path, "test.csv")
            else:
                datapath = os.path.join(self.data_path, "train_unlabeled.csv")

            if not os.path.exists(datapath):
                raise FileNotFoundError(
                    f"Target domain data file not found: {datapath}\n"
                    f"Please ensure data files are in place."
                )
        else:
            # Source domain data loading with fold splits
            if self.is_labeled:
                datapath = os.path.join(self.data_path, "folds", self.setting, f"{self.mode}_fold{self.fold}.csv")
            else:
                datapath = os.path.join(self.data_path, "folds", self.setting, f"train_unlabeled_fold{self.fold}.csv") 

        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Data file not found: {datapath}")

        df = pd.read_csv(datapath)

        # Create SMILES to index mapping
        smiles_to_idx = {smiles: idx for idx, smiles in enumerate(self.smiles_list)}

        # Required columns check
        required_cols = ['SMILES', 'uniprot_id1', 'uniprot_id2']
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        # Filter invalid entries
        valid_mask = (
            df['SMILES'].isin(smiles_to_idx.keys()) &
            df['uniprot_id1'].notna() & (df['uniprot_id1'] != 'na') &
            df['uniprot_id2'].notna() & (df['uniprot_id2'] != 'na') &
            df['uniprot_id1'].isin(self.uniprot_id_list) &
            df['uniprot_id2'].isin(self.uniprot_id_list)
        )
        df_filtered = df[valid_mask]
        
        self.molecule_index_list = df_filtered['SMILES'].tolist()
        self.protein_index1_list = df_filtered['uniprot_id1'].tolist()
        self.protein_index2_list = df_filtered['uniprot_id2'].tolist()

        # Labels (-1 for unlabeled mode)
        if self.is_labeled:
            if 'label' not in df.columns:
                raise ValueError("'label' column required for labeled mode")
            self.label_list = torch.LongTensor(df_filtered['label'].tolist())
        else:
            self.label_list = torch.LongTensor([-1] * len(df_filtered))

        # Map to indices in self.smiles_list
        self.mol_indices = [smiles_to_idx[sm] for sm in self.molecule_index_list]

    def generate_ecfp(self, mol, radius=2, nBits=1024):
        """Generate ECFP fingerprint"""
        if mol is None:
            return np.zeros(nBits, dtype=np.float32)
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            arr = np.zeros((nBits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32)
        except Exception as e:
            print(f"Error generating fingerprint: {e}")
            return np.zeros(nBits, dtype=np.float32)

    def process_molecule(self):
        """Process molecular data (ECFP fingerprints + UniMol2 representations)"""
        compound_path = os.path.join(self.data_path, 'features', 'compound_phy.tsv')
        if not os.path.exists(compound_path):
            raise FileNotFoundError(f"Molecular feature file not found: {compound_path}")

        compounds_df = pd.read_csv(compound_path, sep='\t')
        self.smiles_list = compounds_df.iloc[:, 0].dropna().tolist()

        valid_smiles_list = []
        fingerprints_list = []

        for smi in self.smiles_list:
            mol = AllChem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles_list.append(smi)
                fp = self.generate_ecfp(mol)
                fingerprints_list.append(fp)

        self.smiles_list = valid_smiles_list
        self.fingerprints = torch.from_numpy(np.array(fingerprints_list))  # [N, nBits]

        # Load UniMol2 .pt file if exists
        unimol_pt_path = os.path.join(self.data_path, "features", "smiles_representations.pt")
        if os.path.exists(unimol_pt_path):
            # unimol_data_dict structure: {smiles -> {'cls_repr': np.array, 'atomic_reprs': np.array}}
            self.unimol_data_dict = torch.load(unimol_pt_path)
        else:
            print(f"Warning: {unimol_pt_path} not found. UniMol data not loaded.")
            self.unimol_data_dict = {}

    def process_protein(self):
        """Process protein data"""
        esm2_path = os.path.join(self.data_path, "features", "protein_esm2.csv")
        pfeature_path = os.path.join(self.data_path, "features", "protein_phy.csv")
        protrans_path = os.path.join(self.data_path, "features", "protein_protrans.csv")

        for p in [esm2_path, pfeature_path, protrans_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Protein feature file not found: {p}")

        ESM2 = pd.read_csv(esm2_path, header=None).rename(columns={0: "uniprot_id"})
        pfeature = pd.read_csv(pfeature_path)
        protrans = pd.read_csv(protrans_path)

        # Remove NA
        ESM2.dropna(subset=['uniprot_id'], inplace=True)
        pfeature.dropna(subset=['uniprot_id'], inplace=True)
        protrans.dropna(subset=['uniprot_id'], inplace=True)

        # Get common proteins
        common_proteins = set(ESM2['uniprot_id']) & set(pfeature['uniprot_id']) & set(protrans['uniprot_id'])

        ESM2 = ESM2[ESM2['uniprot_id'].isin(common_proteins)]
        pfeature = pfeature[pfeature['uniprot_id'].isin(common_proteins)]
        protrans = protrans[protrans['uniprot_id'].isin(common_proteins)]

        # Sort by uniprot_id
        ESM2.sort_values('uniprot_id', inplace=True)
        pfeature.sort_values('uniprot_id', inplace=True)
        protrans.sort_values('uniprot_id', inplace=True)

        self.uniprot_id_list = ESM2['uniprot_id'].tolist()

        # Concatenate protein features
        protein_feats = np.concatenate([
            ESM2.iloc[:, 1:].values,
            pfeature.iloc[:, 1:].values,
            protrans.iloc[:, 1:].values
        ], axis=1)
        self.protein_feats_list = torch.FloatTensor(protein_feats)

    def __getitem__(self, idx):
        """Get item by index"""
        # Get basic data
        fingerprint = self.fingerprints[self.mol_indices[idx]]
        p1_idx = self.uniprot_id_list.index(self.protein_index1_list[idx])
        p2_idx = self.uniprot_id_list.index(self.protein_index2_list[idx])
        prot1 = self.protein_feats_list[p1_idx]
        prot2 = self.protein_feats_list[p2_idx]
        protein_feats = torch.cat([prot1, prot2], dim=0)
        label = self.label_list[idx]
        
        # UniMol2 representations
        smiles = self.molecule_index_list[idx]
        unimol_repr = self.unimol_data_dict.get(smiles, None)
        if unimol_repr is None:
            cls_repr = torch.zeros(768)
            atomic_reprs = torch.zeros((1, 768))
        else:
            cls_repr = torch.from_numpy(unimol_repr['cls_repr']).float()
            atomic_reprs = torch.from_numpy(unimol_repr['atomic_reprs']).float()
            if cls_repr.dim() == 2 and cls_repr.size(0) == 1:
                cls_repr = cls_repr.squeeze(0)

        # Return domain label only when using domain adaptation
        if self.use_domain_adaptation:
            domain_label = 1.0 if self.domain == 'source' else 0.0
            domain_label = torch.tensor(domain_label, dtype=torch.float)
            return fingerprint, protein_feats, cls_repr, atomic_reprs, label
        else:
            return fingerprint, protein_feats, cls_repr, atomic_reprs, label

    def __len__(self):
        """Return dataset length"""
        return len(self.label_list)


def collate_modulator_fn(batch, use_domain_adaptation=False):
    """Collate function for DataLoader"""
    if use_domain_adaptation:
        fingerprints_list, protein_list, cls_list, atomic_list, label_list, domain_labels = zip(*batch)
    else:
        fingerprints_list, protein_list, cls_list, atomic_list, label_list = zip(*batch)

    # Process basic features
    fingerprints = torch.stack(fingerprints_list, dim=0)
    protein_feats = torch.stack(protein_list, dim=0)
    cls_repr = torch.stack(cls_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long)

    # Process atomic_reprs with padding
    lengths = [arr.shape[0] for arr in atomic_list]
    max_len = max(lengths) if len(lengths) else 1
    B = len(batch)
    hidden_dim = atomic_list[0].shape[1] if B > 0 else 768

    padded_atomic = torch.zeros((B, max_len, hidden_dim), dtype=torch.float32)
    for i, arr in enumerate(atomic_list):
        n = arr.shape[0]
        padded_atomic[i, :n, :] = arr

    if use_domain_adaptation:
        domain_labels = torch.tensor(domain_labels, dtype=torch.float)
        return fingerprints, protein_feats, cls_repr, padded_atomic, labels, domain_labels
    else:
        return fingerprints, protein_feats, cls_repr, padded_atomic, labels
    
    