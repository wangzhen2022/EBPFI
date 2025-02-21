import matplotlib.pyplot as plt
import numpy as np

# 数据
protocols = ["ARP", "DNS", "ICMP", "Modbus", "NBNS", "NTP", "S7comm", "SMB", "TCP", "UDP"]
x_labels = [500, 1000, 1500, 2000]

f1_values = {
    "ARP": [0.59, 0.58, 0.58, 0.58],
    "DNS": [0.92, 0.92, 0.92, 0.92],
    "ICMP": [0.71, 0.76, 0.76, 0.76],
    "Modbus": [0.90, 0.90, 0.90, 0.90],
    "NBNS": [0.92, 0.92, 0.92, 0.92],
    "NTP": [0.55, 0.56, 0.56, 0.56],
    "S7comm": [0.55, 0.55, 0.58, 0.63],
    "SMB": [0.45, 0.45, 0.50, 0.53],
    "TCP": [0.66, 0.67, 0.66, 0.67],
    "UDP": [0.87, 0.89, 0.90, 0.91]
}

tm_values = {
    "ARP": [0.04, 0.09, 0.11, 0.15],
    "DNS": [0.03, 0.08, 0.11, 0.15],
    "ICMP": [0.03, 0.08, 0.11, 0.15],
    "Modbus": [0.03, 0.08, 0.11, 0.15],
    "NBNS": [0.03, 0.08, 0.11, 0.15],
    "NTP": [0.04, 0.09, 0.11, 0.15],
    "S7comm": [0.03, 0.09, 0.11, 0.15],
    "SMB": [0.04, 0.09, 0.11, 0.15],
    "TCP": [0.04, 0.09, 0.11, 0.15],
    "UDP": [0.04, 0.09, 0.11, 0.15]
}

avg_tm_values = np.mean(list(tm_values.values()), axis=0)

# 颜色映射
colors = plt.cm.get_cmap("tab10", len(protocols))

# 绘制 F1 图
plt.figure(figsize=(8, 5))
for i, protocol in enumerate(protocols):
    plt.plot(x_labels, f1_values[protocol], marker='o', linestyle='-', label=protocol, color=colors(i))
plt.xlabel("F1 Score Across Different Protocols")
plt.ylabel("F1 Score")
plt.xticks(x_labels)  # 设置横坐标只显示指定数值
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
plt.grid()
plt.savefig("F1 Score.png", dpi=300, bbox_inches='tight')
plt.show()

# 绘制平均 Tm 图
plt.figure(figsize=(8, 5))
plt.plot(x_labels, avg_tm_values, marker='o', linestyle='-', label="Average Tm", color='Green')
plt.xlabel("Average Tm Value Across Different Protocols")
plt.ylabel("Average Tm Value")
plt.xticks(x_labels)  # 设置横坐标只显示指定数值
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
plt.grid()
plt.savefig("Average Tm Value.png", dpi=300, bbox_inches='tight')
plt.show()