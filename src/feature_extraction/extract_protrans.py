from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import csv
import os
import pandas as pd
from tqdm import tqdm  # 引入tqdm库，用于进度条显示
import gc  # 垃圾回收

def save_global_features(global_embeddings, filename):
    """
    Save global embeddings to a single CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["UniProt_ID"] + [f"Feature_{i+1}" for i in range(global_embeddings[0][1].shape[0])])
        for uniprot_id, features in global_embeddings:
            writer.writerow([uniprot_id] + features.tolist())

# 设置设备（GPU或CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载Tokenizer和模型
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# 使用全精度模式（适用于CPU）
if device == torch.device("cpu"):
    model.to(torch.float32)

# 读取输入CSV文件
input_file = 'extracted_uniprot_sequences.csv'
output_file = "output/protrans_embeddings1.csv"
batch_output_dir = "output/batches"

# 设置最大序列长度和批量大小
max_length = 1024
batch_size = 1  # 最小批量，减少显存占用

# 读取数据
input_df = pd.read_csv(input_file)
global_embeddings = []

# 创建存储中间批量结果的目录
os.makedirs(batch_output_dir, exist_ok=True)

# 批量处理序列，添加进度条
for i in tqdm(range(0, len(input_df), batch_size), desc="Processing Batches"):
    batch_df = input_df.iloc[i:i+batch_size]
    
    sequences = batch_df['Sequence'].tolist()
    uniprot_ids = batch_df['UniProt_ID'].tolist()
    
    # 替换非标准残基并截断序列
    sequences = [
        " ".join(list(re.sub(r"[UZOB]", "X", seq[:max_length])))  # 替换非标准氨基酸并截断
        for seq in sequences
    ]
    
    # Tokenize输入
    ids = tokenizer(sequences, add_special_tokens=True, padding="longest", truncation=True, max_length=max_length)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # 特征提取
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # 处理批量数据
    for j, seq in enumerate(sequences):
        seq_len = min(len(batch_df['Sequence'].iloc[j]), max_length)
        emb = embedding_repr.last_hidden_state[j, :seq_len]
        global_feature = emb.mean(dim=0)
        global_embeddings.append((uniprot_ids[j], global_feature.cpu().numpy()))

    # 每批次保存一次结果
    batch_file = os.path.join(batch_output_dir, f"batch_{i//batch_size}.csv")
    save_global_features(global_embeddings, batch_file)
    global_embeddings = []  # 清空内存中的全局特征
    
    # 清理缓存，释放显存
    del embedding_repr, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

# 合并所有批次结果为最终文件
all_batches = []
for batch_file in tqdm(os.listdir(batch_output_dir), desc="Merging Batches"):
    if batch_file.endswith(".csv"):
        batch_df = pd.read_csv(os.path.join(batch_output_dir, batch_file))
        all_batches.append(batch_df)
final_df = pd.concat(all_batches, ignore_index=True)
final_df.to_csv(output_file, index=False)

print(f"Feature extraction completed! Results saved to {output_file}")