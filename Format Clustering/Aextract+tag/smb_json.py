import json
import csv
from collections import Counter
import math

# Load JSON data from the file
with open('./JSON-原先/smb.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

smb_traffic = []

packet_count = 0

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# Extract SMB traffic
for packet in data:
    layers = packet['_source']['layers']
    if 'smb' in layers:
        smb_data = layers['smb_raw'][0]

        smb_length = len(smb_data) // 2

        # Calculate entropy
        entropy = calculate_entropy(smb_data)
        
        # Determine label (always 7 for SMB traffic)
        smb_label_code = 4

        smb_traffic.append({
            'hex_data': smb_data,
            'length': smb_length,
            'entropy': entropy,
            'label_code': smb_label_code,
        })

        # Update packet counter
        packet_count += 1
        
        # Check if the packet count reaches 2000, if so, stop extraction
        if packet_count >= 2000:
            break

# Create CSV file: SMB traffic data
with open('./extract-方便ET-BERT/smb/smb_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in smb_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total SMB packets processed:", packet_count)
