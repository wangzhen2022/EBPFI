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
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

import sys
sys.stdout = open('./result1/cont.txt', 'w')
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

def load_protocol_data(protocol_name, base_path='./JSON/', num_rows=2000):
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

    train_data = train_data.reshape(train_data.shape[0], 1, 64, 64)
    train_data = np.repeat(train_data, 3, axis=1)
    train_dataset = TensorDataset(torch.Tensor(train_data).type(torch.float32), torch.Tensor(train_labels).type(torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last = True)

    test_data = test_data.reshape(test_data.shape[0], 1, 64, 64)
    test_data = np.repeat(test_data, 3, axis=1)
    test_dataset = TensorDataset(torch.Tensor(test_data).type(torch.float32), torch.Tensor(test_labels).type(torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, drop_last = True)

    class BoundaryLoss(nn.Module):
        def __init__(self):
            super(BoundaryLoss, self).__init__()
       
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        def forward(self, outputs, targets, input_data):
            batch_size = outputs.size(0)
            num_cols = outputs.size(2)           
            
            binary_mask = targets.unsqueeze(1).expand(-1, num_cols, -1)

            valid_outputs = outputs * binary_mask
            input_data = input_data.mean(dim=1, keepdim=True) 
            valid_input_data = input_data.squeeze(1) * binary_mask 
            
            loss = 0.0
            count = 0
            for i in range(batch_size):
                for j in range(num_cols):
                    if targets[i,j] == 1:
                        count += 1
                    
                        sim = self.cosine_similarity(valid_outputs[i, :, j], valid_input_data[i, :, j])
                        loss += 1 - sim  # 余弦相似度越大，越接近1，loss越小

            # 如果存在目标为1的列，计算平均损失
            if count > 0:
                loss /= count
            else:
                loss = torch.tensor(0.0, requires_grad=True)

            return loss
   
    # 使用DeepLabv3模型
    model = DeepLabv3Plus(num_classes=64, backbone='resnet50').to(device)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    switch_epoch = num_epochs // 2 
    loss_count = []

    Boundary_Loss = BoundaryLoss()

#在初期阶段使用BCEWithLogitsLoss来快速收敛到一个好的分类性能，然后在后期阶段切换到CIoULoss来细化边界框的预测。
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        train_outputs = []
        train_labels = []
        
        for z, (input_data, labels) in enumerate(train_loader):
            input_data = input_data.to(device, torch.float)
            labels = labels.to(device, torch.float)
            outputs, outputs1 = model(input_data)
            

                
            # supervised_loss = criterion(outputs, labels)        
            ciou_loss_value = Boundary_Loss(outputs1, labels, input_data)
            loss = ciou_loss_value 
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_predictions += labels.numel()
            correct_predictions += (predicted == labels).sum().item()
            train_outputs.append(outputs)
            train_labels.append(labels)
            
            if z % 20 == 19:
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, z + 1, running_loss / 20))
                loss_count.append(running_loss / 20)
                running_loss = 0.0  
        

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
        

        
        all_outputs_np = all_outputs.cpu().numpy()
        df = pd.DataFrame(all_outputs_np)
        df.to_csv('./result1/'+str(i) + 'all_outputs.csv', index=False, header=False)