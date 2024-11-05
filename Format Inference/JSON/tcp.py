import json
import numpy as np
import pandas as pd

def load_tcp_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    tcp_data = []
    for packet in data:
        layers = packet['_source']['layers']
  
        if 'tcp' in layers:

            tcp_raw_data = layers['tcp_raw'][0]
            decimal_stream = [int(tcp_raw_data[i:i+2], 16) for i in range(0, len(tcp_raw_data), 2)]
            # Adjust the length to 64 bytes
            if len(decimal_stream) > 64:
                decimal_stream = decimal_stream[:64]
            else:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            tcp_data.append(decimal_stream)
    
    return tcp_data

def create_labels(num_packets, field_starts):
    labels = []
    for _ in range(num_packets):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

field_starts = [0,2,4, 8,12,14,16,18,20]

json_file = './tcp.json'  # Replace with the path to your JSON file
tcp_data = load_tcp_data(json_file)
labels = create_labels(len(tcp_data), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(tcp_data)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./tcp_data.csv', index=False, header=False)
labels_df.to_csv('./tcp_labels.csv', index=False, header=False)
