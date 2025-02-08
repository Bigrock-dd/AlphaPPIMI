from Pfeature import pfeature
import pandas as pd
from tqdm import tqdm
import os
import re

# 输入和输出文件路径
input_csv = "extracted_uniprot_sequences.csv"  # 输入文件路径
output_csv = "physicochemical_features_cleaned.csv"  # 输出文件路径
failed_csv = "failed_sequences_pcp.csv"  # 记录处理失败的序列

# 创建存储临时文件的目录
temp_dir = "temp_sequences"
os.makedirs(temp_dir, exist_ok=True)

# 加载输入 CSV 文件
data = pd.read_csv(input_csv)

# 确保包含必要的列
assert 'UniProt_ID' in data.columns and 'Sequence' in data.columns, "CSV 文件必须包含 UniProt_ID 和 Sequence 列"

# 定义正则表达式，仅保留标准氨基酸字符
valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_pattern = re.compile(f"[^{valid_amino_acids}]")  # 匹配非标准氨基酸

# 存储特征结果和失败的序列
features = []
failed_sequences = []

# 遍历每个序列并计算特征
for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing Sequences"):
    protein_id = row['UniProt_ID']
    sequence = row['Sequence']

    # 替换非标准氨基酸为 ''（移除非法字符）
    cleaned_sequence = aa_pattern.sub('', sequence)

    # 如果清洗后的序列为空，记录失败
    if len(cleaned_sequence) == 0:
        print(f"Sequence for {protein_id} is invalid after cleaning.")
        failed_sequences.append({'UniProt_ID': protein_id, 'Error': 'Empty sequence after cleaning'})
        continue

    # 临时文件路径
    temp_file = os.path.join(temp_dir, f"{protein_id}.txt")

    # 保存清洗后的序列到临时文件
    try:
        with open(temp_file, "w") as f:
            f.write(cleaned_sequence + "\n")

        # 计算物理化学特征
        result = pfeature.pcp(temp_file)

        # 检查返回结果是否为嵌套列表
        if isinstance(result, list) and len(result) == 2:
            # 将特征名和特征值组合为字典
            feature_names = result[0]
            feature_values = result[1]
            feature_dict = dict(zip(feature_names, feature_values))
            feature_dict['UniProt_ID'] = protein_id  # 添加 UniProt_ID
            features.append(feature_dict)
        else:
            print(f"Unexpected format for {protein_id}")
            failed_sequences.append({'UniProt_ID': protein_id, 'Error': 'Unexpected format'})
    except Exception as e:
        print(f"Error processing {protein_id}: {e}")
        failed_sequences.append({'UniProt_ID': protein_id, 'Error': str(e)})

    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)

# 将特征结果保存到输出 CSV 文件
if features:
    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False)
    print(f"Physicochemical features saved to {output_csv}")

# 保存处理失败的序列到 CSV 文件
if failed_sequences:
    failed_df = pd.DataFrame(failed_sequences)
    failed_df.to_csv(failed_csv, index=False)
    print(f"Failed sequences saved to {failed_csv}")

# 清理临时目录
os.rmdir(temp_dir)

# 最终统计结果
print(f"Original records: {len(data)}")
print(f"Features generated: {len(features)}")
print(f"Failed sequences: {len(failed_sequences)}")