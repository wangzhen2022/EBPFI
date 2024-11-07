import json
import csv
from collections import Counter
import math

with open('./JSON/ntp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


ntp_traffic = []
ntp_v3_count = 0
ntp_v4_count = 0

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy
packet_count = 0
# Extract NTP traffic
for packet in data:
    layers = packet['_source']['layers']
    if 'ntp' in layers:
        ntp_version = layers['ntp']['ntp.flags_tree']['ntp.flags.vn']
        ntp_data = layers['ntp_raw'][0]

        ntp_length = len(ntp_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(ntp_data)
        
        # Determine label based on NTP version
        if ntp_version == '3':
            label_code = 0
            ntp_v3_count += 1
        elif ntp_version == '4':
            label_code = 1
            ntp_v4_count += 1
        else:
            continue

        ntp_traffic.append({
            'hex_data': ntp_data,
            'length': ntp_length,
            'entropy': entropy,
            'label_code': label_code,
        })

        # Update packet counter
        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break
            
# Create CSV file: NTP traffic data
with open('./extract-方便ET-BERT/ntp/ntp_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in ntp_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

# Print counts
print("NTP v3 count:", ntp_v3_count)
print("NTP v4 count:", ntp_v4_count)
