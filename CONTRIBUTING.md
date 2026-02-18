# 🤝 Contributing to Federated-IDS

Thank you for your interest in contributing to this project! This guide will help you get started.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Reporting Issues](#reporting-issues)

---

## 📜 Code of Conduct

By participating in this project, you agree to maintain a respectful, inclusive, and constructive environment for all contributors.

---

## 🛠️ How to Contribute

### Types of Contributions Welcome

- 🐛 **Bug fixes** — Fix issues in the federated training pipeline, preprocessing, or evaluation
- ✨ **New features** — Additional aggregation strategies, new attack simulations, new datasets
- 📊 **Experiments** — New ablation studies, benchmark comparisons
- 📝 **Documentation** — Improve README, add comments, update workflow docs
- 🧪 **Tests** — Add or improve test coverage

---

## 💻 Development Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Federated-IDS.git
cd Federated-IDS

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create a feature branch
git checkout -b feature/your-feature-name
```

---

## 🔁 Pull Request Process

1. **Create a branch** from `main` with a descriptive name:
   - `feature/add-trimmed-mean-aggregation`
   - `fix/client-dp-noise-bug`
   - `docs/update-workflow-diagram`

2. **Write your code** following the coding standards below.

3. **Test your changes** — make sure existing tests pass and add new ones if needed.

4. **Commit clearly:**
   ```bash
   git commit -m "feat: add Trimmed Mean aggregation strategy"
   git commit -m "fix: resolve PrivacyEngine batch size mismatch"
   git commit -m "docs: update FL workflow diagram"
   ```

5. **Push and open a Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub with a clear description of what you changed and why.

6. **PR Requirements:**
   - ✅ All existing tests pass
   - ✅ New functionality has tests
   - ✅ Code is documented with docstrings
   - ✅ README updated if new features added

---

## 🎨 Coding Standards

### Python Style
- Follow **PEP 8** style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes:

```python
def aggregate_median(updates: list) -> list:
    """
    Performs coordinate-wise median aggregation over client updates.

    Args:
        updates (list): List of numpy arrays from each client.

    Returns:
        list: Aggregated model weights using coordinate-wise median.
    """
```

### File Organization
- **Preprocessing scripts** → `preprocessing/`
- **Model definitions** → `models/`
- **FL client/server code** → `federated/`
- **Evaluation scripts** → `evaluation/`
- **Result figures** → `results/`

---

## 🐛 Reporting Issues

When reporting a bug, please include:

1. **Environment**: Python version, OS, PyTorch version
2. **Steps to reproduce**: Exact commands that cause the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error message**: Full traceback if applicable

Open an issue at: [https://github.com/Sheshan-swarith/Federated-IDS/issues](https://github.com/Sheshan-swarith/Federated-IDS/issues)

---

## 💡 Feature Requests

Have an idea? Open an issue with the label `enhancement` and describe:
- What problem the feature solves
- How you envision it working
- Any relevant papers or references

---

Thank you for helping make Federated-IDS better! 🔐
