# 🔄 Federated Learning Workflow — Technical Reference

This document provides an in-depth explanation of the complete FL-IDS pipeline, from raw data to final intrusion detection decision.

---

## 📌 Overview

The system follows a **5-stage federated learning pipeline**:

```
Stage 1: Data Preprocessing & Non-IID Split
          ↓
Stage 2: Local Client Training (with Differential Privacy)
          ↓
Stage 3: Robust Server Aggregation (Median / FedProx)
          ↓
Stage 4: Adversarial Attack Simulation
          ↓
Stage 5: Evaluation & Visualization
```

---

## Stage 1: Data Preprocessing & Non-IID Distribution

### Datasets Used

| Dataset | Records | Features | Attack Types | Year |
|---------|---------|----------|--------------|------|
| NSL-KDD | 148,517 | 41 | DoS, Probe, R2L, U2R | 2009 |
| UNSW-NB15 | 257,673 | 49 | Fuzzers, DoS, Exploits, Backdoor + 5 more | 2015 |

### Preprocessing Steps

1. **Load CSV** → drop ID/irrelevant columns
2. **Encode categoricals** → `LabelEncoder` per column
3. **Normalize** → `StandardScaler` fit on train, applied to test
4. **Binarize labels** → Normal (0) vs Attack (1)

### Non-IID Partitioning

Real IoT networks are heterogeneous — different devices see different traffic. We simulate this with **Dirichlet distribution** (α = 0.5):

```
Total Dataset
     │
     ├── Client 1: 70% DoS, 20% Normal, 10% Probe
     ├── Client 2: 80% Normal, 15% R2L, 5% DoS
     ├── Client 3: 60% Probe, 30% Normal, 10% U2R
     └── ...
```

Lower α = more heterogeneous. α = 0.5 is the standard used in FL literature.

---

## Stage 2: Local Client Training with Differential Privacy

### Deep Neural Network Architecture

```
Input (41 features)
    │
    ├── Linear(41 → 128) + ReLU + Dropout(0.3)
    │
    ├── Linear(128 → 64) + ReLU + Dropout(0.3)
    │
    ├── Linear(64 → 32) + ReLU
    │
    └── Linear(32 → 2) → Softmax → [Normal, Attack]
```

### Differential Privacy via Opacus

Each client wraps its optimizer with `PrivacyEngine`:

```
Local Gradient
      │
      ▼
Gradient Clipping (max_grad_norm = 1.0)
      │
      ▼
Gaussian Noise Addition (σ = noise_multiplier × max_grad_norm)
      │
      ▼
Noisy Gradient Update
      │
      ▼
Privacy Budget Tracker (ε, δ = 1e-5)
```

**Privacy-Accuracy Tradeoff:**

| noise_multiplier | ε Budget | Accuracy Impact |
|-----------------|----------|-----------------|
| 0.5 | ~10 | -1% |
| 1.1 | ~4 | -3% ✅ |
| 1.5 | ~1.5 | -8% |

---

## Stage 3: Server Aggregation (FedProx + Median Defense)

### FedProx vs FedAvg

Standard FedAvg simply averages client weights. In non-IID settings this causes **client drift** (local models diverge). FedProx adds a proximal term to each client's loss:

```
L_FedProx = L_local + (μ/2) × ||w - w_global||²
```

This penalizes clients that drift too far from the global model. Result: faster convergence and smaller accuracy gap vs centralized training.

### Coordinate-wise Median Aggregation

Instead of averaging gradients (vulnerable to poisoning), the server takes the **element-wise median** across all client updates:

```
w_global[i] = median(w_client1[i], w_client2[i], ..., w_clientN[i])
```

Why median? A single outlier (poisoned gradient) cannot pull the median far from the true value — unlike the mean.

```
Clients:  [0.5, 0.48, 0.51, 99.9 ← poisoned, 0.49]
Mean:      20.28  ← completely corrupted
Median:    0.50   ← correct and robust ✅
```

---

## Stage 4: Adversarial Attack Simulation

### Attack Type 1: Label-Flipping

Malicious clients flip attack labels (1) to normal (0) during local training, causing the global model to misclassify attacks:

```python
# Malicious client flips labels
if self.is_malicious:
    y_batch = 1 - y_batch   # flip: 0→1, 1→0
```

### Attack Type 2: Model Poisoning

Malicious clients scale their gradient updates by a large factor before sending:

```python
# Amplify gradients to corrupt aggregation
if self.is_malicious:
    params = [p * 10.0 for p in params]
```

### Defense Effectiveness

```
Attack Scenario          | FedAvg (No Defense) | Median Defense
─────────────────────────┼────────────────────-┼──────────────
No attack (baseline)     | 85.0%               | 88.2%
10% label-flipping       | 81.2%               | 87.5%
20% label-flipping       | 74.3%               | 86.1%
30% label-flipping       | 63.8%               | 84.2%
20% model poisoning      | 71.1%               | 85.6%
```

---

## Stage 5: Evaluation & Visualization

### Metrics Computed

- **Accuracy** — overall correct classifications
- **Precision** — of flagged intrusions, how many were real?
- **Recall** — of real intrusions, how many were caught?
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — TP, FP, FN, TN breakdown
- **Privacy Budget (ε)** — formal DP guarantee per round

### Output Figures

| Figure | Description |
|--------|-------------|
| Figure_1.png | Accuracy vs Communication Rounds (FL vs Centralized) |
| Figure_2.png | Confusion Matrix — FL Model (Final Round) |
| Figure_3.png | Robustness: Accuracy vs % Malicious Clients |
| Figure_4.png | Privacy-Accuracy Tradeoff (3 DP settings) |

---

## 🔮 Future Work

- **Secure Aggregation** — encrypt gradients before sending to server
- **Byzantine-resilient aggregation** — Krum, Bulyan beyond Median
- **Communication-efficient FL** — gradient compression, quantization
- **Real IoT deployment** — Raspberry Pi client simulation
- **Heterogeneous model architectures** — different DNN sizes per client

---

*For implementation details, see `federated/client.py` and `federated/server.py`.*
