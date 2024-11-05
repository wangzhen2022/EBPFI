import json
import numpy as np
import pandas as pd

def load_icmp_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    echo_requests = []
    for packet in data:
        layers = packet['_source']['layers']
  
        if 'icmp' in layers:
  
            icmp_type = layers['icmp'].get('icmp.type')
         
            if icmp_type == "8":  # Type 8 is Echo Request
                icmp_raw_data = layers['icmp_raw'][0]  # Extract the raw ICMP data
                decimal_stream = [int(icmp_raw_data[i:i+2], 16) for i in range(0, len(icmp_raw_data), 2)]
                # Adjust the length to 64 bytes
                if len(decimal_stream) > 64:
                    decimal_stream = decimal_stream[:64]
                else:
                    decimal_stream.extend([0] * (64 - len(decimal_stream)))
                echo_requests.append(decimal_stream)
    
    return echo_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

field_starts = [0, 1, 2, 4 ,6,8]

json_file = './icmp.json'  # Replace with the path to your JSON file
echo_requests = load_icmp_data(json_file)
labels = create_labels(len(echo_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(echo_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./icmp_data.csv', index=False, header=False)
labels_df.to_csv('./icmp_labels.csv', index=False, header=False)
