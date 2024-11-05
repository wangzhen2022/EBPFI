import json
import csv
import math
from collections import Counter

# Load JSON data from the file
with open('./JSON-原先/arp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

arp_traffic = []

packet_count = 0

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# Extract ARP traffic
for packet in data:
    layers = packet['_source']['layers']
    if 'arp' in layers:
        arp_data = layers['arp_raw'][0]

        arp_length = len(arp_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(arp_data)
        
        # Label all ARP traffic with 0
        arp_label_code = 0

        arp_traffic.append({
            'hex_data': arp_data,
            'length': arp_length,
            'entropy': entropy,
            'label_code': arp_label_code,
        })

        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break

# Create CSV file: ARP traffic data
with open('./extract-方便ET-BERT/arp/arp_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in arp_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total ARP packets processed:", packet_count)
