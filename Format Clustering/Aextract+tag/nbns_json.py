import json
import csv
import os
from collections import Counter
import math


# 计算熵的函数
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

# 提取NBNS流量的函数
def extract_nbns_traffic(file_path, nbns_traffic, max_messages=2000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    message_count = 0
      
    for packet in data:
        if message_count >= max_messages:
            break

        layers = packet['_source']['layers']
        if 'nbns' in layers:
            nbns = layers['nbns']
            
            # 判断是否为Name Query或Refresh报文类型
            if 'Additional records' in nbns:
                label = 6  # Refresh报文类型
            else:
                label = 5  # Name Query报文类型
            
            # 提取nbns的原始数据
            if 'nbns_raw' in layers:
                nbns_raw_data = layers['nbns_raw'][0]
                nbns_length = len(nbns_raw_data) // 2
                entropy = calculate_entropy(nbns_raw_data)
                
                nbns_traffic.append({
                    'hex_data': nbns_raw_data,
                    'length': nbns_length,
                    'entropy': entropy,
                    'label_code': label
                })
                message_count += 1
                if message_count >= max_messages:
                    break

    return message_count


# 处理单个JSON文件
def process_file(file_path, nbns_traffic, max_messages=2000):
    return extract_nbns_traffic(file_path, nbns_traffic, max_messages)

# 写入CSV文件
def write_to_csv(output_csv_path, nbns_traffic):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hex Data', 'Length', 'Entropy', 'Label'])
        for traffic in nbns_traffic:
            writer.writerow([traffic['hex_data'], traffic['length'], traffic['entropy'], traffic['label_code']])

# 主程序
file_paths = ['./JSON/nbns.json', './JSON/nbns1.json', './JSON/nbns2.json']  # 存储JSON文件的路径
output_csv_path = './extract-方便ET-BERT/nbns/nbns_traffic.csv'  # 输出CSV文件的路径

nbns_traffic = []
max_messages = 2000

# 确保输出目录存在
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# 处理每个文件
total_messages_extracted = 0
for file_path in file_paths:
    if total_messages_extracted < max_messages:
        messages_extracted = process_file(file_path, nbns_traffic, max_messages - total_messages_extracted)
        total_messages_extracted += messages_extracted
    else:
        break

# 写入CSV文件
write_to_csv(output_csv_path, nbns_traffic)

print("NBNS流量数据提取和处理完成。")
