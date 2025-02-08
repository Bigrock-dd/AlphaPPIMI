from Pfeature import pfeature
import pandas as pd
from tqdm import tqdm
import os
import re

input_csv = "extracted_uniprot_sequences.csv"  
output_csv = "physicochemical_features_cleaned.csv"  
failed_csv = "failed_sequences_pcp.csv"  

temp_dir = "temp_sequences"
os.makedirs(temp_dir, exist_ok=True)

data = pd.read_csv(input_csv)

assert 'UniProt_ID' in data.columns and 'Sequence' in data.columns, 
valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_pattern = re.compile(f"[^{valid_amino_acids}]") 

features = []
failed_sequences = []

for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing Sequences"):
    protein_id = row['UniProt_ID']
    sequence = row['Sequence']

    cleaned_sequence = aa_pattern.sub('', sequence)

    if len(cleaned_sequence) == 0:
        print(f"Sequence for {protein_id} is invalid after cleaning.")
        failed_sequences.append({'UniProt_ID': protein_id, 'Error': 'Empty sequence after cleaning'})
        continue

    temp_file = os.path.join(temp_dir, f"{protein_id}.txt")

    try:
        with open(temp_file, "w") as f:
            f.write(cleaned_sequence + "\n")

        result = pfeature.pcp(temp_file)

        if isinstance(result, list) and len(result) == 2:
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

    if os.path.exists(temp_file):
        os.remove(temp_file)

if features:
    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False)
    print(f"Physicochemical features saved to {output_csv}")

if failed_sequences:
    failed_df = pd.DataFrame(failed_sequences)
    failed_df.to_csv(failed_csv, index=False)
    print(f"Failed sequences saved to {failed_csv}")

os.rmdir(temp_dir)

# 最终统计结果
print(f"Original records: {len(data)}")
print(f"Features generated: {len(features)}")
print(f"Failed sequences: {len(failed_sequences)}")
