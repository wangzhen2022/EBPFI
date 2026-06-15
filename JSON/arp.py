import json
import numpy as np
import pandas as pd

def load_arp_data(json_file):
    
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    arp_requests = []

    for packet in data:
        layers = packet['_source']['layers']
  
        if 'arp' in layers:  
            arp_raw_data = layers['arp_raw'][0] 
            decimal_stream = [int(arp_raw_data[i:i+2], 16) for i in range(0, len(arp_raw_data), 2)]
           
            if len(decimal_stream) > 64:
                decimal_stream = decimal_stream[:64]
            else:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            arp_requests.append(decimal_stream)
    
    return arp_requests

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

# Example field start positions for ARP Request
# Assuming the fields start at positions: 0 (Hardware Type), 2 (Protocol Type), 4 (Hardware Addr Length), 5 (Protocol Addr Length),
# 6 (Opcode), 8 (Sender Hardware Address), 14 (Sender Protocol Address), 18 (Target Hardware Address), 24 (Target Protocol Address)
field_starts = [0, 2, 4, 5, 6, 8, 14, 18, 24, 28]
# [0, 1, 2, 4 ,6,8]
# [0, 1, 2, 3, 8, 12, 16, 24, 32, 40, 48]
# [0, 2, 4, 6, 7, 8]
# [0, 2, 4, 6, 8]
# [0, 2, 4, 6, 8, 10, 12]
# [0, 1, 2, 4, 6, 8, 10, 12]
json_file = './arp.json'  # Replace with the path to your JSON file
arp_requests = load_arp_data(json_file)
labels = create_labels(len(arp_requests), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(arp_requests)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./arp_data.csv', index=False, header=False)
labels_df.to_csv('./arp_labels.csv', index=False, header=False)
