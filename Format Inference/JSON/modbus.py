import json
import numpy as np
import pandas as pd

def load_modbus_tcp_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    queries = []
    for packet in data:
        layers = packet['_source']['layers']
  
        if 'modbus' in layers:
            modbus_data = layers['mbtcp_raw'][0]  # Extract the raw Modbus data
            decimal_stream = [int(modbus_data[i:i+2], 16) for i in range(0, len(modbus_data), 2)]
            # Adjust the length to 64 bytes
            if len(decimal_stream) > 64:
                decimal_stream = decimal_stream[:64]
            else:
                decimal_stream.extend([0] * (64 - len(decimal_stream)))
            queries.append(decimal_stream)
    
    return queries

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels

# Example field start positions for Modbus/TCP Query
# Assuming the fields start at positions: 0 (Transaction ID), 2 (Protocol ID), 4 (Length), 6 (Unit ID), 7 (Function Code)
field_starts = [0, 2, 4, 6, 7, 8]

json_file = './modbus.json'  # Replace with the path to your JSON file
queries = load_modbus_tcp_data(json_file)
labels = create_labels(len(queries), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(queries)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./modbus_data.csv', index=False, header=False)
labels_df.to_csv('./modbus_labels.csv', index=False, header=False)
