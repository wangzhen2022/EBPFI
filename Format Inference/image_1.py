import pandas as pd
import matplotlib.pyplot as plt
import os

# 文件路径
file_paths = [
    ['./result2/0all_outputs.csv'],
    ['./result2/1all_outputs.csv'],
    ['./result2/2all_outputs.csv'],
    ['./result2/3all_outputs.csv'],
    ['./result2/4all_outputs.csv'],
    ['./result2/5all_outputs.csv'],
    ['./result2/6all_outputs.csv'],
    ['./result2/7all_outputs.csv'],
    ['./result2/8all_outputs.csv'],
    ['./result2/9all_outputs.csv']
]

name_list = ['Cross-entropy loss']#, 'Boundary loss', 'Cross-entropy loss' Optimization after addition
highlight_positions_list = [
    [0, 2, 4, 5, 6, 8, 14, 18, 24, 28],
    [0, 2, 4, 6, 8, 10, 12],
    [0, 1, 2, 4, 6, 8],
    [0, 2, 4, 6, 7, 8],
    [0, 2, 4, 6, 8, 10, 12],
    [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48],
    [0, 1, 2, 4, 6, 8, 10, 11, 12],
    [0, 4, 5, 6, 7, 9, 10, 12, 14, 22, 24, 26, 28, 30, 32],
    [0, 2, 4, 8, 12, 14, 16, 18, 20],
    [0, 2, 4, 6, 8]
]

# 对应的标题列表
titles = ['arp', 'dns', 'icmp', 'Modbus', 'nbns', 'ntp', 'S7Comm', 'smb', 'tcp', 'udp']

# 确保保存路径存在的函数
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 循环处理每组数据
for group_idx, (file_group, highlight_positions, title) in enumerate(zip(file_paths, highlight_positions_list, titles)):
    for file_path, name in zip(file_group, name_list):
        data = pd.read_csv(file_path, header=None)

        # 复制行数直到2000行
        data = pd.concat([data] * (2000 // len(data) + 1), ignore_index=True).iloc[:2000]

        # 计算每列中1的数量
        sums = data.sum()

        # 画柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(range(0, 64), sums, color='blue')
        plt.xlabel('(b) ' + name, fontsize=16)
        plt.ylabel('Counts', fontsize=16)
        # plt.title(title)
        plt.xticks(range(0, 64, 2))  # 显示每隔一个位置的标签以提高可读性
        plt.grid(True)

        # 标记特定位置的竖状网格线为红色
        for pos in highlight_positions:
            plt.axvline(x=pos, color='red', linestyle='--')

     # 保存图片，按组号命名
        plt.savefig(f'./result2/{group_idx}_{name}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
