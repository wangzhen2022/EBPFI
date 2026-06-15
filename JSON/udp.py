import json
import numpy as np
import pandas as pd

def load_udp_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    udp_requests = []
    for packet in data:
        layers = packet['_source']['layers']
        if 'udp' in layers:
            udp_raw_data = layers['udp_raw'][0]  # Extract the raw UDP data
            # udp_payload = layers['udp']['udp.payload_raw'][0]
            # udp_raw_data = udp_raw_data + udp_payload

            # Convert hex data to decimal
            decimal_stream = [int(udp_raw_data[i:i+2], 16) for i in range(0, len(udp_raw_data), 2)]
            # Ensure length is 64 bytes, padding with zeros if necessary
            if len(decimal_stream) < 64:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            else:
                decimal_stream = decimal_stream[:64]
            udp_requests.append(decimal_stream)
    return udp_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

# UDP报文各字段的起始位置
# 例如：Source Port (2 bytes), Destination Port (2 bytes), Length (2 bytes), Checksum (2 bytes), Data (variable)
field_starts = [0, 2, 4, 6, 8]

json_file = './udp.json'
udp_requests = load_udp_data(json_file)
labels = create_labels(len(udp_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(udp_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./udp_data.csv', index=False, header=False)
labels_df.to_csv('./udp_labels.csv', index=False, header=False)
