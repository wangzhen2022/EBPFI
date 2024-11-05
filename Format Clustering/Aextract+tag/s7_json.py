import json
import csv
from collections import Counter
import math

# Load JSON data from the file
with open('./JSON-原先/s7comm.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

s7comm_traffic = []
s7comm_labels = {
    1: 2,  # JOB
    3: 3,  # ACK_DATA
}

# Function to calculate entropy
def calculate_entropy(hex_data):
    byte_freq = Counter(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    total_bytes = len(hex_data) // 2
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_freq.values())
    return entropy

for packet in data:  
    layers = packet['_source']['layers']  
    if 's7comm' in layers:  
        header = layers['s7comm']['s7comm.header']  
        rosctr = int(header.get('s7comm.header.rosctr', 0))  # 获取ROSCTR并将其转换为整数  
          
        # 根据ROSCTR确定标签代码  
        if rosctr == 3:  
            s7comm_label_code = s7comm_labels[3]  # ACK_DATA  
        elif rosctr == 1:  # 注意：这里假设1是一个具体的请求类型，但在S7comm中可能不是  
            s7comm_label_code = s7comm_labels[1]  # 这里可能是一个错误，因为1通常不是ACK_DATA的ROSCTR  
            # 如果1是另一个请求类型，您可能需要一个不同的标签代码  
        else:  
            s7comm_label_code = -1  # 未知类型  
          
        # 根据ROSCTR确定方向（这里只是一个基于请求/响应的示例）  
        # direction = 'Request' if rosctr % 2 == 0 else 'Response'  
        # 或者，如果您想要'0'和'1'作为方向值：  
        direction = '0' if rosctr % 2 == 0 else '1'  
  
        s7comm_data = layers['s7comm_raw'][0]  
        s7comm_length = len(s7comm_data) // 2  
  
        # Calculate entropy  
        entropy = calculate_entropy(s7comm_data)  
  
        s7comm_traffic.append({  
            'hex_data': s7comm_data,  
            'direction': direction,  # 使用上面定义的direction变量  
            'length': s7comm_length,  
            'entropy': entropy,  
            'label': s7comm_label_code,  
        }) 

# Create CSV file: S7COMM traffic data
with open('./extract-方便ET-BERT/s7comm/s7comm_traffic.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hex Data', 'Direction', 'Length', 'Entropy', 'Label'])
    for traffic in s7comm_traffic:
        writer.writerow([traffic['hex_data'], traffic['direction'], traffic['length'], traffic['entropy'], traffic['label']])

