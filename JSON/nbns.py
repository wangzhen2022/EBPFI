import json
import numpy as np
import pandas as pd

def load_nbns_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    nbns_requests = []
    for packet in data:
        layers = packet['_source']['layers']
        if 'nbns' in layers and 'nbns.flags' in layers['nbns'] and layers['nbns']['nbns.flags_tree']['nbns.flags.opcode'] == "0":
            nbns_raw_data = layers['nbns_raw'][0]  # Extract the raw NBNS data
            # Convert hex data to decimal
            decimal_stream = [int(nbns_raw_data[i:i+2], 16) for i in range(0, len(nbns_raw_data), 2)]
            # Ensure length is 64 bytes, padding with zeros if necessary
            if len(decimal_stream) < 64:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            else:
                decimal_stream = decimal_stream[:64]
            nbns_requests.append(decimal_stream)
    return nbns_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

# 假设field_starts根据NBNS协议的字段来定义
# 例如：Transaction ID (2 bytes), Flags (2 bytes), Questions (2 bytes), Answer RRs (2 bytes), Authority RRs (2 bytes), Additional RRs (2 bytes), Queries (variable), etc.
field_starts = [0, 2, 4, 6, 8, 10, 12]

json_file = './nbns.json'
nbns_requests = load_nbns_data(json_file)
labels = create_labels(len(nbns_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(nbns_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./nbns_data.csv', index=False, header=False)
labels_df.to_csv('./nbns_labels.csv', index=False, header=False)
