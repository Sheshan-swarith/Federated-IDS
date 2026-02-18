# 🔐 Robust Federated Deep Learning-Based Intrusion Detection for Heterogeneous IoT Networks

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-FL%20Framework-pink.svg)](https://flower.dev)
[![Opacus](https://img.shields.io/badge/Opacus-Differential%20Privacy-purple.svg)](https://opacus.ai)
[![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD%20%7C%20UNSW--NB15-orange.svg)](https://www.unb.ca/cic/datasets/nsl.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Intelligent Federated Intrusion Detection Platform

A **privacy-preserving federated learning-based Intrusion Detection System (FL-IDS)** designed for heterogeneous IoT environments. The system trains distributed deep learning models across multiple simulated clients **without sharing raw network traffic data**, combining **robust aggregation** with **differential privacy** to detect intrusions under adversarial conditions.

This project demonstrates an end-to-end federated IDS pipeline — from non-IID data distribution and local deep learning training to Byzantine-resilient aggregation and formal privacy guarantees.

---

## 📊 System Architecture Overview

### High-Level Architecture

```
graph TB
    subgraph "IoT Network Layer"
        C1[IoT Client 1] --> AGG[Federated Aggregation Server]
        C2[IoT Client 2] --> AGG
        C3[IoT Client N] --> AGG
    end

    subgraph "Training Pipeline"
        AGG --> PROX[FedProx Aggregation]
        PROX --> GLOBAL[Global IDS Model]
    end

    subgraph "Privacy & Security Layer"
        DP[Differential Privacy - Opacus] --> C1
        DP --> C2
        DP --> C3
        ATTACK[Adversarial Clients - Label Flipping / Model Poison] --> AGG
        MEDIAN[Median Defense] --> PROX
    end

    subgraph "Evaluation"
        GLOBAL --> EVAL[Accuracy / F1 / Robustness Metrics]
        EVAL --> VIZ[Confusion Matrix & Charts]
    end

    style AGG fill:#f3e5f5,stroke:#4a148c
    style GLOBAL fill:#e8f5e8,stroke:#1b5e20
    style DP fill:#fff3e0,stroke:#e65100
    style MEDIAN fill:#e1f5fe,stroke:#01579b
```

---

## 🏗️ Component Architecture

### 1. Data & Preprocessing Layer

```
┌──────────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                  │
├──────────────────────────────────────────────────────┤
│  • NSL-KDD dataset loading & cleaning                │
│  • UNSW-NB15 dataset loading & cleaning              │
│  • Label encoding & StandardScaler normalization     │
│  • Non-IID client data partitioning (Dirichlet)      │
│  • Train / Test split per client                     │
└──────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────┐
│           NON-IID DISTRIBUTION STRATEGY              │
│  • Dirichlet(α=0.5) distribution per label class    │
│  • Simulates real-world heterogeneous IoT devices    │
│  • Each client sees a different traffic profile      │
└──────────────────────────────────────────────────────┘
```

### 2. Federated Training Layer

```
┌──────────────────────────────────────────────────────┐
│           FEDERATED TRAINING FLOW                    │
├──────────────────────────────────────────────────────┤
│  1. Global model initialized on server               │
│  2. Model weights broadcast to all N clients         │
│  3. Each client trains locally (E epochs)            │
│  4. Differential Privacy noise injected via Opacus   │
│  5. Encrypted gradients sent to server               │
│  6. Server runs Median-based robust aggregation      │
│  7. Updated global model broadcast — repeat R rounds │
└──────────────────────────────────────────────────────┘
```

### 3. Hybrid Defense Engine

```
┌──────────────────────────────────────────────────────┐
│         ADVERSARIAL ROBUSTNESS ENGINE                │
├──────────────────────────────────────────────────────┤
│                  ┌──────────────┐                    │
│                  │ Client Update│                    │
│                  │  (Gradient)  │                    │
│                  └──────┬───────┘                    │
│         ┌───────────────┼──────────────┐             │
│         ▼               ▼              ▼             │
│  ┌─────────────┐ ┌────────────┐ ┌──────────────┐    │
│  │Label-Flipping│ │   Model   │ │  Benign      │    │
│  │  (Malicious) │ │ Poisoning │ │  Client      │    │
│  └─────────────┘ └────────────┘ └──────────────┘    │
│         │               │              │             │
│         └───────────────┼──────────────┘             │
│                         ▼                            │
│              ┌────────────────────┐                  │
│              │  Median Aggregation│                  │
│              │  (Robust Defense)  │                  │
│              └────────────────────┘                  │
└──────────────────────────────────────────────────────┘
```

---

## 🔄 FL Training Flow (Sequence)

```
sequenceDiagram
    participant Server as FL Server
    participant C1 as Client 1 (Benign)
    participant C2 as Client 2 (Benign)
    participant CM as Client M (Malicious)
    participant DP as Opacus DP Engine

    loop Each Round r = 1 to R
        Server->>C1: Broadcast Global Weights
        Server->>C2: Broadcast Global Weights
        Server->>CM: Broadcast Global Weights

        C1->>DP: Local Training
        DP-->>C1: Noisy Gradients (ε, δ)

        C2->>DP: Local Training
        DP-->>C2: Noisy Gradients (ε, δ)

        CM->>CM: Label-Flip / Model Poison
        CM-->>Server: Poisoned Update

        C1-->>Server: DP Gradient Update
        C2-->>Server: DP Gradient Update

        Server->>Server: Median Aggregation (Defense)
        Server->>Server: Update Global Model
    end

    Server->>Server: Evaluate on Test Set
    Server->>Server: Log Accuracy / F1 / Robustness
```

---

## 🎯 Fraud Detection Rules & Risk Logic

### FL Decision Pipeline

```
graph LR
    A[Network Traffic] --> B[Feature Extraction]
    B --> C[Local DNN - Client]
    C --> D[DP Noise - Opacus]
    D --> E[Server Aggregation]
    E --> F{Global Model Prediction}

    F -->|Confidence > 0.8| G[🔴 INTRUSION]
    F -->|Confidence 0.5–0.8| H[🟡 SUSPICIOUS]
    F -->|Confidence < 0.5| I[🟢 NORMAL]

    G --> J[Alert + Log]
    H --> K[Flag for Review]
    I --> L[Pass Through]
```

### Privacy Budget Parameters

```
┌──────────────────────────────────────────────────────┐
│          DIFFERENTIAL PRIVACY CONFIGURATION          │
├──────────────────────────────────────────────────────┤
│  Setting        │ noise_multiplier │ ε (epsilon)     │
│─────────────────┼──────────────────┼─────────────────│
│  Low Privacy    │      0.5         │    ~8–10        │
│  Medium Privacy │      1.1         │    ~3–5  ✅ Used │
│  High Privacy   │      1.5         │    ~1–2         │
│─────────────────┼──────────────────┼─────────────────│
│  δ (delta) = 1e-5 across all settings               │
│  max_grad_norm  = 1.0 (gradient clipping)            │
└──────────────────────────────────────────────────────┘
```

---

## 🏆 Model Performance Metrics

### Results on NSL-KDD Dataset

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **FL + FedProx + DP (Ours)** | **88.2%** | **0.88** | **0.87** | **0.88** |
| Centralized DNN (Baseline) | 91.0% | 0.91 | 0.90 | 0.91 |
| FL + FedAvg (No Defense) | 85.0% | 0.84 | 0.83 | 0.84 |
| FL + FedAvg (No DP) | 85.0% | 0.85 | 0.84 | 0.84 |

### Robustness Under Attack

| Malicious Client % | No Defense | Median Defense (Ours) |
|-------------------|------------|----------------------|
| 10% | 81.2% | 87.5% |
| 20% | 74.3% | 86.1% |
| 30% | 63.8% | 84.2% |

*Median aggregation maintains over 84% accuracy even under 30% adversarial clients.*

---

## 📈 Performance & Scalability

```
┌──────────────────────────────────────────────────────┐
│            SYSTEM PERFORMANCE CHARACTERISTICS        │
├──────────────────────────────────────────────────────┤
│  FL Accuracy (NSL-KDD):     88.2%                    │
│  FL Accuracy (UNSW-NB15):   89.1%                    │
│  Centralized Accuracy:      91.0%                    │
│  Accuracy Gap (FL vs. Cen): ~2.8%  ✅               │
│  Privacy Budget (ε):        ~3–5 (Medium Privacy)   │
│  Communication Rounds:      30                       │
│  Clients Simulated:         10                       │
│  Robustness at 30% attack:  84.2%                    │
└──────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Federated-IDS/
├── preprocessing/
│   ├── nslkdd_preprocess.py      # NSL-KDD data loader & cleaner
│   └── unsw_preprocess.py        # UNSW-NB15 data loader & cleaner
├── models/
│   └── dnn_model.py              # Deep Neural Network architecture
├── federated/
│   ├── client.py                 # Flower FL client (DP + local training)
│   └── server.py                 # Flower FL server (FedProx + Median)
├── evaluation/
│   ├── metrics.py                # Accuracy, F1, confusion matrix
│   ├── ablation_clients.py       # Client count ablation
│   └── ablation_attacks.py       # Attack % ablation
├── results/
│   ├── Figure_1.png              # Accuracy vs Rounds
│   ├── Figure_2.png              # Confusion Matrix (FL)
│   ├── Figure_3.png              # Robustness under attack
│   └── Figure_4.png              # Privacy-Accuracy tradeoff
├── data/
│   ├── nslkdd/                   # Place NSL-KDD CSVs here
│   └── unsw/                     # Place UNSW-NB15 CSVs here
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── FEDERATED_LEARNING_WORKFLOW.md
└── TEST_CASES.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Sheshan-swarith/Federated-IDS.git
cd Federated-IDS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

**NSL-KDD:**
```bash
# Download from: https://www.unb.ca/cic/datasets/nsl.html
# Place KDDTrain+.txt and KDDTest+.txt in data/nslkdd/
```

**UNSW-NB15:**
```bash
# Download from: https://www.kaggle.com/datasets/programmer3/unsw-nb15-dataset
# Place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in data/unsw/
```

### 4. Preprocess Data

```bash
python preprocessing/nslkdd_preprocess.py
python preprocessing/unsw_preprocess.py
```

### 5. Run Federated Training

```bash
# Start the FL server
python federated/server.py

# In separate terminals, start clients
python federated/client.py --client-id 0
python federated/client.py --client-id 1
# ... up to N clients
```

### 6. Evaluate Results

```bash
python evaluation/metrics.py
```

---

## 🔒 Security & Privacy Features

- **Differential Privacy**: Formal (ε, δ)-DP guarantees via Opacus PrivacyEngine
- **Gradient Clipping**: max_grad_norm=1.0 prevents gradient leakage
- **Robust Aggregation**: Coordinate-wise Median defense against poisoning
- **No Raw Data Sharing**: Only model gradients exchanged between clients and server
- **Adversarial Simulation**: Label-flipping and model poisoning attack modes

---

## 📊 Monitoring & Analysis

- **Confusion Matrix**: Per-round and final evaluation
- **Accuracy vs Rounds**: Convergence visualization
- **Privacy Budget Tracker**: ε consumed per round
- **Robustness Plots**: Accuracy under varying attack percentages

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Sheshan Swarith**
- GitHub: [@Sheshan-swarith](https://github.com/Sheshan-swarith)
- Project: Final Year — Information Security

---

## 📚 References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
- Li et al., "Federated Optimization in Heterogeneous Networks (FedProx)", MLSys 2020
- Yousefpour et al., "Opacus: User-Friendly Differential Privacy Library in PyTorch", 2021
- Moustafa & Slay, "UNSW-NB15: A Comprehensive Dataset", MilCIS 2015
- Beutel et al., "Flower: A Friendly Federated Learning Framework", 2020
