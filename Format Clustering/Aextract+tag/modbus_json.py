import json
import csv
from collections import Counter
import math

# 加载JSON文件中的数据
with open('./JSON/modbus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

modbus_traffic = []
packet_count = 0

# 计算熵的函数
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# 提取Modbus/TCP流量
for packet in data:
    layers = packet['_source']['layers']
    if 'modbus' in layers:
        # 提取每个包的第一个modbus数据
        modbus_data = layers['mbtcp_raw'][0]
        modbus_length = len(modbus_data) // 2
        entropy = calculate_entropy(modbus_data)
        
        modbus_traffic.append({
            'hex_data': modbus_data,
            'length': modbus_length,
            'entropy': entropy,
            'label_code': 4
        })

        packet_count += 1
        
        # 如果处理的包数达到2000，停止提取
        if packet_count >= 2000:
            break

# 创建CSV文件并写入Modbus/TCP流量数据
with open('./extract-方便ET-BERT/modbus/modbus_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
    for traffic in modbus_traffic:
        writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

print("Total Modbus/TCP packets processed:", packet_count)
