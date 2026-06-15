# Semantic Protocol Format Inference

The project treats protocol payload bytes as 64x64 "images" and predicts field boundaries (start positions) for 10 common network/industrial protocols.

## Supported Protocols

| # | Protocol | Type       |
| - | -------- | ---------- |
| 0 | ARP      | Network    |
| 1 | DNS      | Network    |
| 2 | ICMP     | Network    |
| 3 | Modbus   | Industrial |
| 4 | NBNS     | Network    |
| 5 | NTP      | Network    |
| 6 | S7comm   | Industrial |
| 7 | SMB      | Network    |
| 8 | TCP      | Transport  |
| 9 | UDP      | Transport  |

## Project Structure

```
.
├── model.py                    # model definition
├── main.py                     # Training + inference (leave-one-out)
├── Inferential statistics.py   # Final metric evaluation
├── JSON/
│   ├── {protocol}.py           # Data preprocessing per protocol
│   ├── {protocol}_data.csv     # Processed payload bytes (N × 64)
│   └── {protocol}_labels.csv   # Field boundary ground truth (N × 64)
└── result/                     # Output directory (created at runtime)
    ├── {i}.pth                 # Model weights for fold i
    └── {i}alloutputs.csv       # Predictions for fold i (test protocol)
```

## Pipeline Overview

### 1. Data Preparation (`JSON/{protocol}.py`)

Each protocol has a script that:

- Parses PCAP/JSON traces to extract raw payload bytes
- Converts hex to decimal byte streams
- Pads or truncates to exactly **64 bytes**
- Annotates field boundaries as binary labels (1 = field start)

Output: `{protocol}_data.csv` and `{protocol}_labels.csv`

### 2. Model Definition (`model.py`)

A DeepLabv3+ architecture adapted for 1D byte sequence segmentation:

- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **ASPP**: Atrous Spatial Pyramid Pooling with dilation rates 6, 12, 18
- **Decoder**: Fuses low-level features with ASPP output, followed by **Boundary-Guided Filter (BGF)**
- **Attention**: CBAM (Channel + Spatial Attention) in the boundary filter
- **Output**: Two branches — a 64-class logit vector and a 64×64 feature map used for cosine-similarity-based boundary loss

### 3. Training & Inference (`main.py`)

- **Strategy**: Leave-one-out cross-validation — for each of the 10 protocols, train on the other 9 and test on the held-out protocol
- **Sliding Window**: Each 2000-row CSV is sliced with `window_size=64`, `step_size=32`, producing overlapping 64×64 samples
- **Input**: 64×64 single-channel expanded to 3 channels (RGB-like) for ResNet compatibility
- **Loss**: `BCEWithLogitsLoss` (classification) + cosine-similarity boundary loss on the secondary branch
- **Optimizer**: Adam, lr=0.001, 2 epochs
- **Outputs**:
  - `./result/{i}.pth` — saved model weights for each fold
  - `./result/{i}alloutputs.csv` — per-byte predictions for the test protocol

### 4. Evaluation (`Inferential statistics.py`)

- Reads saved prediction CSVs and ground-truth labels
- Aggregates predictions by taking the **mode** (most frequent value) across all windows for each byte position
- Computes per-protocol **Precision**, **Recall**, and **F1 Score**

## Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- torchmetrics
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies:

```bash
pip install torch torchvision torchmetrics pandas numpy scikit-learn matplotlib
```

## Usage

### Step 1: Prepare Data

Run each protocol's preprocessing script in `JSON/` to generate `_data.csv` and `_labels.csv`:

```bash
cd JSON
python arp.py
python dns.py
# ... run all protocol scripts
cd ..
```

### Step 2: Train & Infer

```bash
python main.py
```

This runs leave-one-out training for all 10 protocols and saves model weights and prediction CSVs to `./result/`.

### Step 3: Compute Metrics

```bash
python Inferential\ statistics.py
```

Outputs Precision, Recall, and F1 Score for each protocol.

## Key Design Choices

- **Image-like representation**: Byte sequences are reshaped as 64×64 matrices, enabling the use of 2D CNNs pretrained on natural images
- **Boundary loss**: A cosine-similarity loss on the secondary branch encourages feature consistency within the same semantic field
- **Leave-one-out**: Tests generalization to unseen protocols — the model must infer field boundaries for a protocol it has never seen during training
