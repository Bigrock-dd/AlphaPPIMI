import pandas as pd
import numpy as np
from tqdm import tqdm
from unimol_tools import UniMolRepr
from rdkit import Chem

# 初始化 UniMol 模型
clf = UniMolRepr(
    data_type='molecule',
    remove_hs=False,
    model_name='unimolv2',  # 可用: unimolv1, unimolv2
    model_size='84m',       # 如果使用 unimolv2，可选: 84m, 164m, 310m, 570m, 1.1B
)

# 读取 SMILES 数据
file_path = "compound_phy.tsv"  # 修改为你的文件路径
data = pd.read_csv(file_path, sep="\t")
smiles_list = data['SMILES'].tolist()

# 验证并过滤非法 SMILES
valid_smiles = []
for smi in tqdm(smiles_list, desc="Validating SMILES"):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        valid_smiles.append(smi)
    else:
        print(f"Invalid SMILES skipped: {smi}")

print(f"Total valid SMILES: {len(valid_smiles)}")

# 输出文件路径
output_file = "smiles_features_fixed.csv"  # 输出的 CSV 文件路径

# 处理所有有效 SMILES
print("Processing all valid SMILES...")
all_features = []

fixed_cls_dim = 768  # 固定 CLS 表示的维度
fixed_atomic_dim = 128  # 固定 atomic_reprs 的维度

try:
    # 提取分子表示
    unimol_repr = clf.get_repr(valid_smiles, return_atomic_reprs=True)

    # 存储结果
    for i, smiles in enumerate(tqdm(valid_smiles, desc="Storing Features")):
        try:
            cls_repr = unimol_repr['cls_repr'][i]
            atomic_reprs = unimol_repr['atomic_reprs'][i]

            # 截断或填充 cls_repr 到固定长度
            if len(cls_repr) > fixed_cls_dim:
                cls_repr = cls_repr[:fixed_cls_dim]
            else:
                cls_repr = np.pad(cls_repr, (0, fixed_cls_dim - len(cls_repr)), constant_values=0)

            # 截断或填充 atomic_reprs 到固定长度
            if len(atomic_reprs) > fixed_atomic_dim:
                atomic_reprs = atomic_reprs[:fixed_atomic_dim]
            else:
                atomic_reprs = np.pad(atomic_reprs, ((0, fixed_atomic_dim - len(atomic_reprs)), (0, 0)), constant_values=0)

            # 将特征存储到 all_features 列表
            all_features.append({
                'SMILES': smiles,
                'cls_repr': ','.join(map(str, cls_repr)),  # 将 cls_repr 转换为字符串
                'atomic_reprs': ';'.join(
                    [','.join(map(str, atom)) for atom in atomic_reprs]
                )  # 将 atomic_reprs 转换为字符串
            })
        except Exception as e:
            print(f"Error storing features for SMILES: {smiles}, Error: {e}")
except Exception as e:
    print(f"Error during feature extraction: {e}")

# 将所有特征保存到 CSV 文件
features_df = pd.DataFrame(all_features)
features_df.to_csv(output_file, index=False)
print(f"All features saved to {output_file}")






