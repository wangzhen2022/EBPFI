import pandas as pd
import matplotlib.pyplot as plt
import os

# 文件路径
file_paths = [
    ['./result2/0all_outputs.csv'],
    # ['./result/1all_outputs.csv'],
    # ['./result/2all_outputs.csv'],
    # ['./result/3all_outputs.csv'],
    # ['./result2/4all_outputs.csv'],
    # ['./result/5all_outputs.csv'],
    # ['./result/6all_outputs.csv'],
    # ['./result/7all_outputs.csv'],
    # ['./result/8all_outputs.csv'],
    # ['./result2/9all_outputs.csv']
]

highlight_positions_list = [
    [0, 2, 4, 5, 6, 8, 14, 18, 24, 28],
    # [0, 2, 4, 6, 8, 10, 12],
    # [0, 1, 2, 4, 6, 8],
    # [0, 2, 4, 6, 7, 8],
    # [0, 2, 4, 6, 8, 10, 12],
    # [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48],
    # [0, 1, 2, 4, 6, 8, 10, 11, 12],
    # [0, 4, 5, 6, 7, 9, 10, 12, 14, 22, 24, 26, 28, 30, 32],
    # [0, 2, 4, 8, 12, 14, 16, 18, 20],
    # [0, 2, 4, 6, 8]
]

name_list = ['ARP_Cross-entropy loss']

# name_list = ['ARP_Dual Loss', 'DNS_Normal Optimization', 
#             'ICMP_Normal Optimization', 'Modbus_Normal Optimization',
#             'NBNS_Normal Optimization', 'NTP_Normal Optimization', 
#             'S7Comm_Normal Optimization', 'SMB_Normal Optimization', 
#             'TCP_Normal Optimization', 'UDP_Dual Loss']


# name_list = ['ARP_Cross-entropy loss', 'DNS_Cross-entropy loss', 
#             'ICMP_Cross-entropy loss', 'Modbus_Cross-entropy loss',
#             'NBNS_Cross-entropy loss', 'NTP_Cross-entropy loss', 
#             'S7Comm_Cross-entropy loss', 'SMB_Cross-entropy loss', 
#             'TCP_Cross-entropy loss', 'UDP_Cross-entropy loss']

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 循环处理每组数据
for group_idx, (file_group, highlight_positions, name) in enumerate(zip(file_paths, highlight_positions_list, name_list)):

    for file_path in file_group:
        data = pd.read_csv(file_path, header=None)

        # 复制行数直到2000行
        data = pd.concat([data] * (2000 // len(data) + 1), ignore_index=True).iloc[:2000]
        sums = data.sum()

        # 画柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(range(0, 64), sums, color='blue')
        plt.xlabel('(b) ' + name , fontsize=16)
        plt.ylabel('Counts', fontsize=16)
        plt.xticks(range(0, 64, 2))  
        plt.grid(True)

        for pos in highlight_positions:
            plt.axvline(x=pos, color='red', linestyle='--')

        # 保存图像
        plt.savefig(f'./result2/{group_idx}_{name}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

