import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torchmetrics
from torch.utils.data import random_split
from model1 import DeepLabv3Plus
import time
import math
import os

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 创建保存热力图的目录
os.makedirs('./heatmaps', exist_ok=True)

def calculate_metrics(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels), "The length of true labels and predicted labels must be the same."
    
    TP = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 1 and pl == 1)
    FP = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 0 and pl == 1)
    FN = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 1 and pl == 0)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def create_dataloader(data, labels):
    window_size = 64
    step_size = 32
    images = []
    image_labels = []
    num_rows = data.shape[0]
    for start in range(0, num_rows - window_size + 1, step_size):
        image = data[start:start + window_size, :]
        images.append(image)
        label = labels[start+1, :]
        image_labels.append(label)
    images = np.array(images)
    image_labels = np.array(image_labels)
    return images, image_labels

def load_protocol_data(protocol_name, base_path='./JSON/', num_rows=1000):
    data = pd.read_csv(f'{base_path}{protocol_name}_data.csv', header=None).iloc[:num_rows,:64].values
    labels = pd.read_csv(f'{base_path}{protocol_name}_labels.csv', header=None).iloc[:num_rows,:64].values
    return data, labels

# 加载协议数据
icmp_data, icmp_labels = load_protocol_data('icmp')
ntp_data, ntp_labels = load_protocol_data('ntp')
smb_data, smb_labels = load_protocol_data('smb')
udp_data, udp_labels = load_protocol_data('udp')
s7comm_data, s7comm_labels = load_protocol_data('s7comm')
nbns_data, nbns_labels = load_protocol_data('nbns')
modbus_data, modbus_labels = load_protocol_data('modbus')
arp_data, arp_labels = load_protocol_data('arp')
dns_data, dns_labels = load_protocol_data('dns')
tcp_data, tcp_labels = load_protocol_data('tcp')

data_labels_pairs = [
    # (arp_data, arp_labels),
    # (dns_data, dns_labels),
    # (icmp_data, icmp_labels),
    # (modbus_data, modbus_labels),
    # (nbns_data, nbns_labels),
    # (ntp_data, ntp_labels),
    # (s7comm_data, s7comm_labels),
    # (smb_data, smb_labels),
    # (tcp_data, tcp_labels),
    (udp_data, udp_labels)
]

train_data_sets = []
train_label_sets = []
test_data_sets = []
test_label_sets = []

for i in range(len(data_labels_pairs)):
    combined_data = []
    combined_labels = []

    data, labels = create_dataloader(data_labels_pairs[i][0], data_labels_pairs[i][1])
    combined_data.append(data)
    combined_labels.append(labels)

train_data = np.concatenate(combined_data, axis=0)
train_labels = np.concatenate(combined_labels, axis=0)

train_data = train_data.reshape(train_data.shape[0], 1, 64, 64)
train_data = np.repeat(train_data, 3, axis=1)
train_dataset = TensorDataset(torch.Tensor(train_data).type(torch.float32), torch.Tensor(train_labels).type(torch.float32))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)


model = DeepLabv3Plus(num_classes=64, backbone='resnet50').to(device)
model.load_state_dict(torch.load('./result/7.pth'), strict=False)

# def save_heatmap(heatmap, index, save_dir="./heatmaps"):
#     os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
#     plt.figure(figsize=(6, 6))
#     plt.imshow(heatmap, cmap='hot', interpolation='nearest')  # 选择热力图颜色
#     plt.colorbar()
#     plt.axis("off")  # 关闭坐标轴
#     plt.savefig(f"{save_dir}/udp_{index}.png", bbox_inches='tight', pad_inches=0.1)
#     plt.close()

def save_heatmap(heatmap, index, field_starts, save_dir="./heatmaps"):
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    plt.figure(figsize=(6, 6))
    
    # 创建热力图
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')  # 选择热力图颜色
    

    # for col in field_starts:
    #     plt.axvline(x=col, color='blue', linestyle='--', linewidth=1.5)
    plt.xlabel('Column', fontsize=10)  # 设置X轴标签字体大小
    plt.ylabel('Row', fontsize=10)     # 设置Y轴标签字体大小
    plt.title('UDP', fontsize=10)      # 设置标题字体大小
    plt.colorbar()
    # plt.axis("off")  # 关闭坐标轴
    plt.savefig(f"{save_dir}/udp_{index}.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

# 处理模型输出并保存热力图
# field_starts = [0, 2, 4, 5, 6, 8, 14, 18, 24, 28]
field_starts = [0, 2, 4, 6, 8]

for batch_idx, (samples, labels1) in enumerate(train_loader):
    samples = samples.to(device, torch.float)
    labels1 = labels1.to(device, torch.float)
    outputs, outputs1 = model(samples)

    for i in range(outputs1.shape[0]):
        heatmap = outputs1[i].detach().cpu().numpy()  # 转换为 numpy 数组
        save_heatmap(heatmap, index=batch_idx * train_loader.batch_size + i, field_starts=field_starts)