import pandas as pd
import sys
import time
sys.stdout = open('./Quantitativeresult/2000output.txt', 'w')

protocols = ["arp", "dns", "icmp", "modbus", "nbns", "ntp", "s7comm", "smb", "tcp", "udp"]

def calculate_metrics(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels), "The length of true labels and predicted labels must be the same." 
    TP = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 1 and pl == 1)
    FP = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 0 and pl == 1)
    FN = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 1 and pl == 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

results = []
    
for i, protocol in enumerate(protocols):
    outputs_path = f"./Quantitativeresult/{i}2000_outputs.csv"  
    labels_path = f"./JSON/{protocol}_labels.csv" 
    start_time = time.time()
    outputs_df = pd.read_csv(outputs_path, header=None)
    final_sequence = outputs_df.apply(lambda col: col.mode()[0]).tolist()
    labels_df = pd.read_csv(labels_path, header=None)
    true_labels = labels_df.iloc[0].tolist()
    precision, recall, f1_score = calculate_metrics(true_labels, final_sequence)
    results.append({
        "protocol": protocol,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    })
    end_time = time.time()
    epoch_time = end_time - start_time
    print('*********', epoch_time)

for result in results:
    print(f"Protocol: {result['protocol']}")
    print(f"  Precision: {result['precision']}")
    print(f"  Recall: {result['recall']}")
    print(f"  F1 Score: {result['f1_score']}")
    print("-" * 30)
