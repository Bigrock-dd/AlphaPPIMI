from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import csv
import os
import pandas as pd
from tqdm import tqdm 
import gc

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

if device == torch.device("cpu"):
    model.to(torch.float32)

input_file = 'extracted_uniprot_sequences.csv'
output_file = "output/protrans_embeddings1.csv"
batch_output_dir = "output/batches"

max_length = 1024
batch_size = 1 

input_df = pd.read_csv(input_file)
global_embeddings = []

os.makedirs(batch_output_dir, exist_ok=True)

for i in tqdm(range(0, len(input_df), batch_size), desc="Processing Batches"):
    batch_df = input_df.iloc[i:i+batch_size]
    
    sequences = batch_df['Sequence'].tolist()
    uniprot_ids = batch_df['UniProt_ID'].tolist()
    
    sequences = [
        " ".join(list(re.sub(r"[UZOB]", "X", seq[:max_length])))  
        for seq in sequences
    ]
    
    ids = tokenizer(sequences, add_special_tokens=True, padding="longest", truncation=True, max_length=max_length)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    for j, seq in enumerate(sequences):
        seq_len = min(len(batch_df['Sequence'].iloc[j]), max_length)
        emb = embedding_repr.last_hidden_state[j, :seq_len]
        global_feature = emb.mean(dim=0)
        global_embeddings.append((uniprot_ids[j], global_feature.cpu().numpy()))

    batch_file = os.path.join(batch_output_dir, f"batch_{i//batch_size}.csv")
    save_global_features(global_embeddings, batch_file)
    global_embeddings = []  
    
    del embedding_repr, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()


all_batches = []
for batch_file in tqdm(os.listdir(batch_output_dir), desc="Merging Batches"):
    if batch_file.endswith(".csv"):
        batch_df = pd.read_csv(os.path.join(batch_output_dir, batch_file))
        all_batches.append(batch_df)
final_df = pd.concat(all_batches, ignore_index=True)
final_df.to_csv(output_file, index=False)

print(f"Feature extraction completed! Results saved to {output_file}")
