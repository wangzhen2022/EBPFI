import json
import numpy as np
import pandas as pd

def load_smb_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    smb_requests = []
    for packet in data:
        layers = packet['_source']['layers']
  
        if 'smb' in layers:
            smb_raw_data = layers['smb_raw'][0]  # Extract the raw SMB data
            decimal_stream = [int(smb_raw_data[i:i+2], 16) for i in range(0, len(smb_raw_data), 2)]
            # Adjust the length to 64 bytes
            if len(decimal_stream) > 64:
                decimal_stream = decimal_stream[:64]
            else:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            smb_requests.append(decimal_stream)
    
    return smb_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

# Define the field starts specific to SMB
field_starts = [0, 4, 5, 6, 7, 9, 10, 12, 14, 22, 24, 26, 28, 30, 32]  # Modify according to SMB fields

json_file = './smb.json'  # Replace with the path to your JSON file
smb_requests = load_smb_data(json_file)
labels = create_labels(len(smb_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(smb_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./smb_data.csv', index=False, header=False)
labels_df.to_csv('./smb_labels.csv', index=False, header=False)
