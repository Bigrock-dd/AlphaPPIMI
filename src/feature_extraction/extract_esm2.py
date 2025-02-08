import torch
import esm
import pandas as pd
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()


input_csv = "extracted_uniprot_sequences.csv"  
output_csv = "protein_esm2_features_cleaned.csv"  
failed_csv = "failed_sequences.csv"  
invalid_csv = "invalid_sequences.csv"  


data = pd.read_csv(input_csv)
print(f"Original records: {len(data)}")


data = data.dropna(subset=['UniProt_ID', 'Sequence'])
data['Sequence'] = data['Sequence'].astype(str).str.strip()  
data = data[data['Sequence'].str.len() > 0]  
data = data.drop_duplicates(subset=['UniProt_ID'])  
print(f"Cleaned records: {len(data)}")


MAX_LENGTH = 1024
data['Truncated_Sequence'] = data['Sequence'].str[:MAX_LENGTH]
data['Truncated_Length'] = data['Truncated_Sequence'].str.len()

invalid_sequences = data[data['Truncated_Length'] < 1]
print(f"Invalid truncated sequences (length < 1): {len(invalid_sequences)}")

if not invalid_sequences.empty:
    invalid_sequences.to_csv(invalid_csv, index=False)
    print(f"Invalid sequences saved to {invalid_csv}")

valid_data = data[data['Truncated_Length'] > 0]
sequences = [(row['UniProt_ID'], row['Truncated_Sequence']) for _, row in valid_data.iterrows()]
print(f"Valid sequences after cleaning and truncation: {len(sequences)}")


batch_size = 1  
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


features_df = pd.DataFrame(sequence_representations)
features_df.insert(0, 'UniProt_ID', protein_ids)
features_df.to_csv(output_csv, index=False)
print(f"Protein features saved to {output_csv}")

if failed_sequences:
    failed_df = pd.DataFrame(failed_sequences, columns=["Failed_UniProt_ID"])
    failed_df.to_csv(failed_csv, index=False)
    print(f"Failed sequences saved to {failed_csv}")

print(f"Original records: {len(data)}")
print(f"Valid sequences processed: {len(sequences)}")
print(f"Features generated: {len(sequence_representations)}")
print(f"Failed sequences: {len(failed_sequences)}")
