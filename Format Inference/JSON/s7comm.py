import json
import numpy as np
import pandas as pd
#ACK_DATA
def load_s7comm_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    ack_data_packets = []
    for packet in data:
        layers = packet['_source']['layers']
        if 's7comm' in layers:
            s7comm_data = layers['s7comm']
            if 's7comm.header' in s7comm_data:
                header = s7comm_data['s7comm.header']
                
                if header.get('s7comm.header.rosctr') == '3':  # 3 indicates an Ack_Data
              
            
                    s7comm_raw_data = layers['s7comm_raw'][0]  # Extract the raw S7comm data
             
                    decimal_stream = [int(s7comm_raw_data[i:i+2], 16) for i in range(0, len(s7comm_raw_data), 2)]
                    # Adjust the length to 64 bytes
                    if len(decimal_stream) > 64:
                        decimal_stream = decimal_stream[:64]
                    else:
                        decimal_stream.extend([0] * (64 - len(decimal_stream)))
                    ack_data_packets.append(decimal_stream)
    return ack_data_packets

def create_labels(num_requests, field_starts):
    labels = []
    for _ in range(num_requests):
        label = [1 if i in field_starts else 0 for i in range(64)]
        labels.append(label)
    return labels


field_starts = [0, 1, 2, 4, 6, 8, 10, 11, 12]

json_file = './s7comm.json'  # Path to your JSON file
ack_data_packets = load_s7comm_data(json_file)
labels = create_labels(len(ack_data_packets), field_starts)

# Convert to numpy arrays and save as CSV
data = np.array(ack_data_packets)
labels = np.array(labels)

data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)
data_df.to_csv('./s7comm_data.csv', index=False, header=False)
labels_df.to_csv('./s7comm_labels.csv', index=False, header=False)
