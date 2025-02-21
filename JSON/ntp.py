import json
import numpy as np
import pandas as pd

def load_ntp_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    ntp_requests = []
    for packet in data:
        layers = packet['_source']['layers']
        if 'ntp' in layers and 'ntp.flags_tree' in layers['ntp'] and layers['ntp']['ntp.flags_tree']['ntp.flags.vn'] == "3":
            ntp_raw_data = layers['ntp_raw'][0]  # Extract the raw NTP data
            # Convert hex data to decimal
            decimal_stream = [int(ntp_raw_data[i:i+2], 16) for i in range(0, len(ntp_raw_data), 2)]
            # Ensure length is 48 bytes for NTPv3 standard length, padding with zeros if necessary
            if len(decimal_stream) < 64:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            else:
                decimal_stream = decimal_stream[:64]
            ntp_requests.append(decimal_stream)
    return ntp_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

field_starts = [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48]

json_file = './ntp.json'
ntp_requests = load_ntp_data(json_file)
labels = create_labels(len(ntp_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(ntp_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./ntp_data.csv', index=False, header=False)
labels_df.to_csv('./ntp_labels.csv', index=False, header=False)
