import json
import csv
from collections import Counter
import math

# Load JSON data from the file
with open('./JSON-原先/dns.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dns_traffic = []

packet_count = 0

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# Extract DNS query traffic
for packet in data:
    layers = packet['_source']['layers']
    if 'dns' in layers:
        dns_data = layers['dns_raw'][0]

        dns_length = len(dns_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(dns_data)
        
        # Determine label (always 1 for DNS query)
        dns_label_code = 1

        dns_traffic.append({
            'hex_data': dns_data,
            'length': dns_length,
            'entropy': entropy,
            'label_code': dns_label_code,
        })

        # Update packet counter
        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break

# Create CSV file: DNS traffic data
with open('./extract-方便ET-BERT/dns/dns_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in dns_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total DNS query packets processed:", packet_count)
