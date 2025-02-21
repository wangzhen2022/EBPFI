import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载CSV文件
file_path = './JSON/arp_data.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path, header=None)

# 定义步长和窗口大小
step_size = 5
window_size = 64

# 创建保存图像的目录
output_dir = './image/arp_data_images'
os.makedirs(output_dir, exist_ok=True)

# 定义函数来创建并保存图像
def create_images(data, window_size, step_size, output_dir):
    num_rows = data.shape[0]
    img_count = 0

    for start in range(0, num_rows - window_size + 1, step_size):
        window_data = data.iloc[start:start + window_size].values

        # 绘制图像
        plt.figure(figsize=(6, 6))
        plt.imshow(window_data, cmap=plt.cm.viridis, aspect='auto', vmin=0, vmax=255)
        plt.colorbar()

        # plt.xlabel('Column')
        # plt.ylabel('Row')
        # plt.title('ARP')
        plt.xlabel('Column', fontsize=10)  # 设置X轴标签字体大小
        plt.ylabel('Row', fontsize=10)     # 设置Y轴标签字体大小
        plt.title('ARP', fontsize=10)      # 设置标题字体大小

        # 设置 x 轴刻度
        plt.xticks(range(0, window_size, 4))
        plt.gca().invert_yaxis()
        # 设置 y 轴刻度，确保从下到上逐渐变大
        # plt.yticks(range(0, window_size, 8))

        # 保存图像
        plt.savefig(f'{output_dir}/img_{img_count}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        img_count += 1
      

# 生成图像
create_images(data, window_size, step_size, output_dir)