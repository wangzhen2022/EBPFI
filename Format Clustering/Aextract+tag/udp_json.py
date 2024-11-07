import json
import csv
from collections import Counter
import math

# Load JSON data from the file
with open('./JSON/udp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

udp_traffic = []

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# Extract UDP traffic
packet_count = 0
for packet in data:
    layers = packet['_source']['layers']
    if 'udp' in layers:
        udp_data = layers['udp_raw'][0]
        udp_data = layers['udp_raw'][0]  # Extract the raw UDP data
        udp_payload = layers['udp']['udp.payload_raw'][0]
        udp_data = udp_data + udp_payload
        udp_length = len(udp_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(udp_data)

        udp_traffic.append({
            'hex_data': udp_data,
            'length': udp_length,
            'entropy': entropy,
            'label_code': 6,
        })

        # Update packet counter
        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break
        
# Create CSV file: UDP traffic data
with open('./extract-方便ET-BERT/udp/udp_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in udp_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total UDP packets processed:", packet_count)
