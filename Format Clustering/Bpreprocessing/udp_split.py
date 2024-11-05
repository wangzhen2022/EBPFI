import pandas as pd
from sklearn.model_selection import train_test_split
import os


udp_data = pd.read_csv('../Aextract+tag/extract-方便ET-BERT/udp/udp_traffic.csv')
hex_data = udp_data['Hex Data']

clip_data_split = hex_data.apply(lambda x: ' '.join([x[i:i+2] for i in range(0, len(x), 2)]))

if len(clip_data_split) != len(udp_data):
    raise ValueError("Lengths of clip_data_split and traindata do not match!")

udp_data['Hex Data'] = clip_data_split

data_list = [udp_data]
file_names = ["udp_data"]  # 文件名列表

for data, file_name in zip(data_list, file_names):
    traindata, test = train_test_split(data, test_size=0.1, random_state=42)
    testdata, valdata = train_test_split(test, test_size=0.5, random_state=42)  # Adjusted to 0.5

    traindata = traindata.drop(['Length', 'Entropy'], axis=1)
    new_order = ['Label', 'Hex Data']
    traindata = traindata[new_order]
    traindata = traindata.rename(columns={'Label': 'label', 'Hex Data': 'text_a'})
    
    # output_path = os.path.join('../CET-BERT-main/datasets/udp/', f'train{file_name}.tsv')
    traindata.to_csv('../CET-BERT-main/datasets/udp/train'+ file_name+ '.tsv', sep='\t', index=False)

    valdata = valdata.drop('Length', axis=1)
    valdata = valdata.drop('Entropy', axis=1)
    new_order = ['Label', 'Hex Data']
    valdata = valdata[new_order]
    valdata = valdata.rename(columns={'Label': 'label', 'Hex Data': 'text_a'})
    valdata.to_csv('../CET-BERT-main/datasets/udp/val'+ file_name +'.tsv', sep='\t', index=False)

    testdata = testdata.drop('Length', axis=1)
    testdata = testdata.drop('Entropy', axis=1)
    new_order = ['Label', 'Hex Data']
    testdata = testdata[new_order]
    testdata = testdata.rename(columns={'Label': 'label', 'Hex Data': 'text_a'})
    testdata.to_csv('../CET-BERT-main/datasets/udp/test'+ file_name +'.tsv', sep='\t', index=False)

