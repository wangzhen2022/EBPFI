import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure
from sklearn.mixture import GaussianMixture
from collections import defaultdict
import matplotlib.pyplot as plt
# 读取并合并数据
arp_datafeature = pd.read_csv('./arp/arp_datafeature.csv', header=None)
arp_datalabel = pd.read_csv('./arp/arp_datalabel.csv', header=None)

dns_datafeature = pd.read_csv('./dns/dns_datafeature.csv', header=None)
dns_datalabel = pd.read_csv('./dns/dns_datalabel.csv', header=None)

icmp_datafeature = pd.read_csv('./icmp/icmp_datafeature.csv', header=None)
icmp_datalabel = pd.read_csv('./icmp/icmp_datalabel.csv', header=None)

modbus_datafeature = pd.read_csv('./modbus/modbus_datafeature.csv', header=None)
modbus_datalabel = pd.read_csv('./modbus/modbus_datalabel.csv', header=None)

nbns_datafeature = pd.read_csv('./nbns/nbns_datafeature.csv', header=None)
nbns_datalabel = pd.read_csv('./nbns/nbns_datalabel.csv', header=None)

ntp_datafeature = pd.read_csv('./ntp/ntp_datafeature.csv', header=None)
ntp_datalabel = pd.read_csv('./ntp/ntp_datalabel.csv', header=None)

s7comm_datafeature = pd.read_csv('./s7comm/s7comm_datafeature.csv', header=None)
s7comm_datalabel = pd.read_csv('./s7comm/s7comm_datalabel.csv', header=None)

smb_datafeature = pd.read_csv('./smb/smb_datafeature.csv', header=None)
smb_datalabel = pd.read_csv('./smb/smb_datalabel.csv', header=None)

tcp_datafeature = pd.read_csv('./tcp/tcp_datafeature.csv', header=None)
tcp_datalabel = pd.read_csv('./tcp/tcp_datalabel.csv', header=None)

udp_datafeature = pd.read_csv('./udp/udp_datafeature.csv', header=None)
udp_datalabel = pd.read_csv('./udp/udp_datalabel.csv', header=None)

merged_data = pd.concat([arp_datafeature, dns_datafeature, icmp_datafeature, modbus_datafeature, nbns_datafeature, ntp_datafeature, s7comm_datafeature, smb_datafeature,  tcp_datafeature, udp_datafeature], axis=0)
labels = pd.concat([arp_datalabel,dns_datalabel, icmp_datalabel, modbus_datalabel, nbns_datalabel, ntp_datalabel, s7comm_datalabel,smb_datalabel, tcp_datalabel, udp_datalabel], axis=0)

print(merged_data.shape)
print(labels.shape)
pca = PCA(n_components=0.99)  # 保留 95% 的方差
X_pca = pca.fit_transform(merged_data)
y_true = labels.values.squeeze()

hdbscan_cluster = HDBSCAN(min_cluster_size=15, min_samples=10, gen_min_span_tree=True)
y_pred = hdbscan_cluster.fit_predict(X_pca)
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred, beta=0.1)
# print(f" Homogeneity: {homogeneity}, Completeness: {completeness}, V-Measure: {v_measure}")

plt.scatter(merged_data.iloc[:, 8], merged_data.iloc[:, 9], c=y_true, cmap='rainbow', alpha=0.7)
# plt.title('(a) Format Clustering-Labels')
plt.xlabel('(a) Format Clustering - Labels', fontsize=10)
plt.colorbar(ticks=np.arange(np.unique(y_true).size))
plt.savefig('./cluster_hdbscan1.png')
plt.show()

plt.scatter(merged_data.iloc[:, 8], merged_data.iloc[:, 9], c=y_pred, cmap='rainbow', alpha=0.7)
# plt.title('(b) Format Clustering-Predictions')
plt.xlabel('(b) Format Clustering - Predictions', fontsize=10)
plt.colorbar(ticks=np.arange(-1,np.unique(y_pred).size))
plt.savefig('./cluster_hdbscan2.png')
plt.show()
print(np.unique(y_true).size)
print(np.unique(y_pred).size)
exit()
for i in range(0, 11):
    for j in range(0, 11):
        if i != j:  # 避免画出对角线上的重复图
            plt.figure()
            plt.scatter(merged_data.iloc[:, i], merged_data.iloc[:, j], c=y_pred, cmap='rainbow', alpha=0.7)
            plt.xlabel(merged_data.columns[i])
            plt.ylabel(merged_data.columns[j])
            plt.title(f'Scatter plot of feature {i} vs {j}')
            plt.colorbar(label='Format Clustering-Labels')
            plt.savefig(f'./image/cluster_hdbscan_{i}_{j}.png')
            plt.close()  # 关闭当前图形以释放内存
  