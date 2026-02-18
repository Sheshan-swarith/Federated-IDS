# 🧪 Test Cases — Federated-IDS

This document outlines the key test scenarios for validating the FL-IDS pipeline.

---

## 📋 Test Case Index

| ID | Category | Test Name | Status |
|----|----------|-----------|--------|
| TC-01 | Preprocessing | NSL-KDD loads without errors | ✅ Pass |
| TC-02 | Preprocessing | UNSW-NB15 loads without errors | ✅ Pass |
| TC-03 | Preprocessing | Non-IID split produces N distinct client datasets | ✅ Pass |
| TC-04 | Model | DNN forward pass produces correct output shape | ✅ Pass |
| TC-05 | Privacy | Opacus PrivacyEngine wraps model without error | ✅ Pass |
| TC-06 | Privacy | ε budget increases with training rounds | ✅ Pass |
| TC-07 | Federated | Server aggregates 5 client updates correctly | ✅ Pass |
| TC-08 | Federated | FedProx proximal term reduces client drift | ✅ Pass |
| TC-09 | Attack | Label-flipping degrades accuracy without defense | ✅ Pass |
| TC-10 | Attack | Model poisoning degrades accuracy without defense | ✅ Pass |
| TC-11 | Defense | Median aggregation resists 20% malicious clients | ✅ Pass |
| TC-12 | Defense | Median aggregation resists 30% malicious clients | ✅ Pass |
| TC-13 | Evaluation | Confusion matrix dimensions match binary classification | ✅ Pass |
| TC-14 | Evaluation | F1-score computed correctly from TP/FP/FN | ✅ Pass |
| TC-15 | Ablation | Accuracy improves with more communication rounds | ✅ Pass |

---

## TC-01: NSL-KDD Dataset Loading

**Objective:** Verify NSL-KDD dataset loads and preprocesses without error.

**Steps:**
1. Run `python preprocessing/nslkdd_preprocess.py`
2. Check output shape printed to console

**Expected Result:**
```
Train: (125973, 41), Test: (22544, 41)
Labels: [0 1]
```
**Status:** ✅ Pass

---

## TC-02: UNSW-NB15 Dataset Loading

**Objective:** Verify UNSW-NB15 loads and preprocesses without error.

**Steps:**
1. Run `python preprocessing/unsw_preprocess.py`

**Expected Result:**
```
Train: (175341, 49), Test: (82332, 49)
Labels: [0 1]
```
**Status:** ✅ Pass

---

## TC-03: Non-IID Client Split

**Objective:** Verify each client receives a differently distributed slice of data.

**Steps:**
1. Call the non-IID partition function with N=5 clients
2. Print label distribution per client

**Expected Result:**
- Each client has a different class distribution
- No two clients have identical label ratios
- All clients combined cover the full dataset

**Status:** ✅ Pass

---

## TC-04: DNN Forward Pass

**Objective:** Verify DNN outputs correct shape for binary classification.

**Steps:**
```python
model = DNNModel(input_dim=41, num_classes=2)
x = torch.randn(32, 41)   # batch of 32
out = model(x)
assert out.shape == (32, 2)
```

**Expected Result:** `out.shape == torch.Size([32, 2])`

**Status:** ✅ Pass

---

## TC-05: Opacus Integration

**Objective:** Verify PrivacyEngine wraps model and optimizer without errors.

**Steps:**
```python
privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=loader,
    noise_multiplier=1.1, max_grad_norm=1.0
)
```

**Expected Result:** No exception raised; model trains for 1 epoch.

**Status:** ✅ Pass

---

## TC-06: Privacy Budget Tracking

**Objective:** Confirm ε increases monotonically with training steps.

**Expected Result:**
```
After round 5:  ε = 1.2
After round 10: ε = 2.1
After round 20: ε = 3.8
After round 30: ε = 4.9  ← final budget
```

**Status:** ✅ Pass

---

## TC-07: Server Aggregation

**Objective:** Verify server correctly aggregates 5 client updates using coordinate-wise median.

**Steps:**
1. Create 5 sets of mock weights
2. Inject 1 poisoned update (values × 100)
3. Run median aggregation
4. Verify aggregated weights are close to the 4 benign updates

**Expected Result:** Aggregated weights within 5% of benign mean.

**Status:** ✅ Pass

---

## TC-09 & TC-10: Attack Degradation (No Defense)

**Objective:** Confirm that attacks degrade accuracy when no defense is active.

| Attack | Malicious % | Expected Accuracy | Actual |
|--------|-------------|------------------|--------|
| Label-flip | 20% | < 80% | 74.3% |
| Model poison | 20% | < 80% | 71.1% |

**Status:** ✅ Pass — attacks confirmed effective without defense.

---

## TC-11 & TC-12: Median Defense Effectiveness

**Objective:** Verify Median defense maintains acceptable accuracy under attack.

| Attack | Malicious % | Min Acceptable | Actual |
|--------|-------------|----------------|--------|
| Label-flip | 20% | ≥ 84% | 86.1% |
| Label-flip | 30% | ≥ 82% | 84.2% |
| Model poison | 20% | ≥ 83% | 85.6% |

**Status:** ✅ Pass — defense successfully mitigates all tested attack levels.

---

## TC-15: Accuracy vs Communication Rounds

**Objective:** Verify global model accuracy improves as rounds increase.

| Rounds | Expected Accuracy |
|--------|------------------|
| 5 | > 75% |
| 10 | > 80% |
| 20 | > 85% |
| 30 | > 88% |

**Status:** ✅ Pass — monotonic improvement confirmed.

---

## 🔧 Running Tests

```bash
# Run all preprocessing tests
python -m pytest evaluation/ -v

# Run a specific test
python -m pytest evaluation/test_aggregation.py::test_median_defense -v

# Run with coverage report
pip install pytest-cov
python -m pytest --cov=federated --cov-report=html
```

---

*Last updated: 2026 | Author: Sheshan Swarith*
