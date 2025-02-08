import torch
import esm
import pandas as pd
from tqdm import tqdm

# 加载 ESM 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()

# 文件路径
input_csv = "extracted_uniprot_sequences.csv"  # 输入文件
output_csv = "protein_esm2_features_cleaned.csv"  # 特征文件
failed_csv = "failed_sequences.csv"  # 处理失败的序列
invalid_csv = "invalid_sequences.csv"  # 无效序列文件

# 读取数据
data = pd.read_csv(input_csv)
print(f"Original records: {len(data)}")

# 数据清洗：去除空值、空字符串、重复ID
data = data.dropna(subset=['UniProt_ID', 'Sequence'])
data['Sequence'] = data['Sequence'].astype(str).str.strip()  # 移除两端空格
data = data[data['Sequence'].str.len() > 0]  # 去除空字符串
data = data.drop_duplicates(subset=['UniProt_ID'])  # 去除重复的UniProt_ID
print(f"Cleaned records: {len(data)}")

# 截断序列长度为最大1024
MAX_LENGTH = 1024
data['Truncated_Sequence'] = data['Sequence'].str[:MAX_LENGTH]
data['Truncated_Length'] = data['Truncated_Sequence'].str.len()

# 筛选截断后长度小于1的序列
invalid_sequences = data[data['Truncated_Length'] < 1]
print(f"Invalid truncated sequences (length < 1): {len(invalid_sequences)}")

# 保存无效序列到文件
if not invalid_sequences.empty:
    invalid_sequences.to_csv(invalid_csv, index=False)
    print(f"Invalid sequences saved to {invalid_csv}")

# 准备有效序列数据
valid_data = data[data['Truncated_Length'] > 0]
sequences = [(row['UniProt_ID'], row['Truncated_Sequence']) for _, row in valid_data.iterrows()]
print(f"Valid sequences after cleaning and truncation: {len(sequences)}")

# 批量处理序列并捕获错误
batch_size = 1  # 根据显存调整批量大小
sequence_representations = []
protein_ids = []
failed_sequences = []

for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches"):
    batch = sequences[i:i + batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)

    try:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30])
        token_representations = results["representations"][30]

        for j, (_, seq) in enumerate(batch):
            sequence_rep = token_representations[j, 1: len(seq) + 1].mean(0)
            sequence_representations.append(sequence_rep.cpu().numpy())
            protein_ids.append(batch[j][0])
    except Exception as e:
        print(f"Failed to process sequence {batch[0][0]}: {e}")
        failed_sequences.append(batch[0][0])

# 保存成功提取的特征到 CSV 文件
features_df = pd.DataFrame(sequence_representations)
features_df.insert(0, 'UniProt_ID', protein_ids)
features_df.to_csv(output_csv, index=False)
print(f"Protein features saved to {output_csv}")

# 保存失败的序列 ID 到文件
if failed_sequences:
    failed_df = pd.DataFrame(failed_sequences, columns=["Failed_UniProt_ID"])
    failed_df.to_csv(failed_csv, index=False)
    print(f"Failed sequences saved to {failed_csv}")

# 最终统计结果
print(f"Original records: {len(data)}")
print(f"Valid sequences processed: {len(sequences)}")
print(f"Features generated: {len(sequence_representations)}")
print(f"Failed sequences: {len(failed_sequences)}")