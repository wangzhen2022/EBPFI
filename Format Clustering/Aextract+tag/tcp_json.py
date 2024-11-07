import json
import csv
from collections import Counter
import math

# Load JSON data from the file
with open('./JSON/tcp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tcp_traffic = []

packet_count = 0

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# Extract TCP traffic
for packet in data:
    layers = packet['_source']['layers']
    if 'tcp' in layers:
        tcp_data = layers['tcp_raw'][0]

        tcp_length = len(tcp_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(tcp_data)
        
        # Determine label (always 8 for TCP traffic)
        tcp_label_code = 5

        tcp_traffic.append({
            'hex_data': tcp_data,
            'length': tcp_length,
            'entropy': entropy,
            'label_code': tcp_label_code,
        })

        # Update packet counter
        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break

# Create CSV file: TCP traffic data
with open('./extract-方便ET-BERT/tcp/tcp_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in tcp_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total TCP packets processed:", packet_count)
