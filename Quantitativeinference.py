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
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

import sys
# sys.stdout = open('./result/output2.txt', 'w')
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
    (arp_data, arp_labels),
    (dns_data, dns_labels),
    (icmp_data, icmp_labels),
    (modbus_data, modbus_labels),
    (nbns_data, nbns_labels),
    (ntp_data, ntp_labels),
    (s7comm_data, s7comm_labels),
    (smb_data, smb_labels),
    (tcp_data, tcp_labels),
    (udp_data, udp_labels)
]

train_data_sets = []
train_label_sets = []
test_data_sets = []
test_label_sets = []

for i in range(len(data_labels_pairs)):
    combined_data = []
    combined_labels = []
    for j in range(len(data_labels_pairs)):
        if j != i:
            data, labels = create_dataloader(data_labels_pairs[j][0], data_labels_pairs[j][1])
            combined_data.append(data)
            combined_labels.append(labels)

    train_data = np.concatenate(combined_data, axis=0)
    train_labels = np.concatenate(combined_labels, axis=0)
    test_data, test_labels = create_dataloader(data_labels_pairs[i][0], data_labels_pairs[i][1])

    test_data = test_data.reshape(test_data.shape[0], 1, 64, 64)
    test_data = np.repeat(test_data, 3, axis=1)
    test_dataset = TensorDataset(torch.Tensor(test_data).type(torch.float32), torch.Tensor(test_labels).type(torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, drop_last = True)


    model = DeepLabv3Plus(num_classes=64, backbone='resnet50').to(device)
    model.load_state_dict(torch.load('./result/'+str(i) + '.pth'), strict=False)

    acc = torchmetrics.Accuracy(task='binary', num_classes=64).to(device)
    recall = torchmetrics.Recall(task='binary', average='none', num_classes=64).to(device)
    precision = torchmetrics.Precision(task='binary', average='none', num_classes=64).to(device)
    f1 = torchmetrics.F1Score(task='binary', average='none', num_classes=64).to(device)
    from sklearn.metrics import f1_score
    best_threshold = 0.5

    print("test_action")
    all_outputs = []
    all_labels = []
    start_time = time.time()
    with torch.no_grad():
        for samples, labels1 in test_loader:
            samples = samples.to(device, torch.float)
            labels1 = labels1.to(device, torch.float)
            outputs, _ = model(samples)
            all_outputs.append(outputs)
            all_labels.append(labels1)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_outputs = torch.sigmoid(all_outputs)
        all_outputs = (all_outputs > best_threshold).float()    # best_threshold
        
        # Compute metrics using the custom function
        precision, recall, f1_value = calculate_metrics(all_labels.cpu().numpy().flatten(), all_outputs.cpu().numpy().flatten())

        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1_value:.4f}")
        end_time = time.time()
        epoch_time = end_time - start_time
        print('*********', epoch_time)
        
        all_outputs_np = all_outputs.cpu().numpy()
        df = pd.DataFrame(all_outputs_np)
        df.to_csv('./Quantitativeresult/'+str(i) + '2000_outputs.csv', index=False, header=False)