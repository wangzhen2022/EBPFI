import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import homogeneity_completeness_v_measure
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

data_files = [
    '../Aextract+tag/extract1-标签正确/arp/arp_traffic.csv',
    '../Aextract+tag/extract1-标签正确/dns/dns_traffic.csv',
    '../Aextract+tag/extract1-标签正确/icmp/icmp_traffic.csv',
    '../Aextract+tag/extract1-标签正确/modbus/modbus_traffic.csv',
    '../Aextract+tag/extract1-标签正确/nbns/nbns_traffic.csv',
    '../Aextract+tag/extract1-标签正确/ntp/ntp_traffic.csv',
    '../Aextract+tag/extract1-标签正确/s7comm/s7comm_traffic.csv',
    '../Aextract+tag/extract1-标签正确/smb/smb_traffic.csv',
    '../Aextract+tag/extract1-标签正确/tcp/tcp_traffic.csv',
    '../Aextract+tag/extract1-标签正确/udp/udp_traffic.csv'
]

hex_data_list = []
label_list = []
for file in data_files:
    data = pd.read_csv(file)
    hex_data_list.append(data['Hex Data'])
    label_list.append(data['Label'])

merged_data = pd.concat(hex_data_list, axis=0, ignore_index=True)
merged_label = pd.concat(label_list, axis=0, ignore_index=True)

# 将十六进制数据转换为整数数组
def hex_to_int_array(hex_string, max_length=100):
    int_array = np.array([int(ch, 16) for ch in hex_string])
    # 如果长度小于最大长度，填充0
    if len(int_array) < max_length:
        padded_array = np.zeros(max_length)
        padded_array[:len(int_array)] = int_array
        return padded_array
    # 如果长度大于等于最大长度，截断
    else:
        return int_array[:max_length]

# 转换每个十六进制字符串为整数数组，并指定最大长度为100（可根据实际情况调整）
merged_data_numeric = merged_data.apply(hex_to_int_array)

# 创建一个二维数组，每行代表一个样本，每列是一个特征
X = np.array([x for x in merged_data_numeric])

y_true = merged_label.values.squeeze()


hdbscan_cluster = HDBSCAN(min_cluster_size=15, min_samples=10, gen_min_span_tree=True)# min_cluster_size=18, min_samples=10
y_pred = hdbscan_cluster.fit_predict(X)
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred,beta=0.1)
print(f" Homogeneity: {homogeneity}, Completeness: {completeness}, V-Measure: {v_measure}")
_, _, v_measure1 = homogeneity_completeness_v_measure(y_true, y_pred,beta=0.2)
_, _, v_measure2 = homogeneity_completeness_v_measure(y_true, y_pred,beta=0.4)
_, _, v_measure3 = homogeneity_completeness_v_measure(y_true, y_pred,beta=0.7)
print(v_measure1,v_measure2,v_measure3)

# OPTICS clustering
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
clusters_optics = optics.fit_predict(X)
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, clusters_optics, beta=0.1)
print(f"OPTICS - Homogeneity: {homogeneity}, Completeness: {completeness}, V-Measure: {v_measure}")
_, _, v_measure1 = homogeneity_completeness_v_measure(y_true, clusters_optics,beta=0.2)
_, _, v_measure2 = homogeneity_completeness_v_measure(y_true, clusters_optics,beta=0.4)
_, _, v_measure3 = homogeneity_completeness_v_measure(y_true, clusters_optics,beta=0.7)
print(v_measure1,v_measure2,v_measure3)