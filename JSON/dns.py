import json
import numpy as np
import pandas as pd

def load_dns_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    dns_requests = []
    for packet in data:
        layers = packet['_source']['layers']
  
        if 'dns' in layers:
            dns_raw_data = layers['dns_raw'][0]  # Extract the raw DNS data
            decimal_stream = [int(dns_raw_data[i:i+2], 16) for i in range(0, len(dns_raw_data), 2)]
            # Adjust the length to 64 bytes
            if len(decimal_stream) > 64:
                decimal_stream = decimal_stream[:64]
            else:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            dns_requests.append(decimal_stream)
    
    return dns_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

# Define the field starts specific to DNS
field_starts = [0, 2, 4, 6, 8, 10, 12]  # Modify according to DNS fields

json_file = './dns.json'  # Replace with the path to your JSON file
dns_requests = load_dns_data(json_file)
labels = create_labels(len(dns_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(dns_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./dns_data.csv', index=False, header=False)
labels_df.to_csv('./dns_labels.csv', index=False, header=False)
