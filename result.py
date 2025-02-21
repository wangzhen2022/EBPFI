def read_labels_from_csv(file_path, num_lines=None):
    with open(file_path, 'r') as f:
        labels = []
        for i, line in enumerate(f):
            if num_lines is not None and i >= num_lines:
                break
            labels.extend([int(float(value)) for value in line.strip().split(',')])
    return labels

def calculate_metrics(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels),                                                 "The length of true labels and predicted labels must be the same."
    

    TP = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 1 and pl == 1)
    FP = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 0 and pl == 1)
    FN = sum(1 for tl, pl in zip(true_labels, predicted_labels) if tl == 1 and pl == 0)
    

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

protocol_files = [
    ("ARP", './JSON/arp_labels.csv', './result3/0all_outputs.csv'),
    ("DNS", './JSON/dns_labels.csv', './result3/1all_outputs.csv'),
    ("ICMP", './JSON/icmp_labels.csv', './result3/2all_outputs.csv'),
    ("MODBUS", './JSON/modbus_labels.csv', './result3/3all_outputs.csv'),
    ("NBNS", './JSON/nbns_labels.csv', './result3/4all_outputs.csv'),
    ("NTP", './JSON/ntp_labels.csv', './result3/5all_outputs.csv'),
    ("S7COMM", './JSON/s7comm_labels.csv', './result3/6all_outputs.csv'),
    ("SMB", './JSON/smb_labels.csv', './result3/7all_outputs.csv'),
    ("TCP", './JSON/tcp_labels.csv', './result3/8all_outputs.csv'),
    ("UDP", './JSON/udp_labels.csv', './result3/9all_outputs.csv'),
]

# Specify the number of lines to read
num_lines = 960

results = []

for protocol, true_labels_file, predicted_labels_file in protocol_files:
    true_labels = read_labels_from_csv(true_labels_file, num_lines)
    predicted_labels = read_labels_from_csv(predicted_labels_file, num_lines)

    precision, recall, f1_score = calculate_metrics(true_labels, predicted_labels)

    results.append({
        "Protocol": protocol,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    })

    print(f"{protocol} - Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

# Save results to file
import csv

output_file = 'protocol_metrics_results.csv'

with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['Protocol', 'Precision', 'Recall', 'F1-Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results saved to {output_file}")
