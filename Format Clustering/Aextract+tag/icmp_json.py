import json
import csv
from collections import Counter
import math

# Load JSON data from the file
with open('./JSON/icmp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

icmp_traffic = []

icmp_labels = {
    3: {3: 2},  # Destination Unreachable (Port Unreachable)
}

packet_count = 0

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# Extract ICMP traffic
for packet in data:
    layers = packet['_source']['layers']
    if 'icmp' in layers:
        icmp_type = int(layers['icmp']['icmp.type'])
        icmp_code = int(layers['icmp']['icmp.code'])
        icmp_data = layers['icmp_raw'][0]

        icmp_length = len(icmp_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(icmp_data)
        
        # Determine label
        if icmp_type == 3 and icmp_code == 3:
            icmp_label_code = 3
        else:
            icmp_label_code = 2

        icmp_traffic.append({
            'hex_data': icmp_data,
            'length': icmp_length,
            'entropy': entropy,
            'label_code': icmp_label_code,
        })

        # Update packet counter
        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break

# Create CSV file: ICMP traffic data
with open('./extract-方便ET-BERT/icmp/icmp_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in icmp_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total ICMP packets processed:", packet_count)
