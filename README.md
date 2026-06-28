# Neural Network Backpropagation from Scratch

Implementation of backpropagation algorithm from scratch using only NumPy — no TensorFlow, no Keras. Two variants: regression and binary classification.

---

## Files

| File | Task | Activation | Loss |
|------|------|-----------|------|
| `backpropagation_regression_scratch.py` | Regression | Linear (none) | MSE |
| `classification_backpropegation_scratch.py` | Classification | Sigmoid | Binary Cross-Entropy |

---

## Network Architecture

```
Input Layer (2 neurons)  →  Hidden Layer (2 neurons)  →  Output Layer (1 neuron)
   cgpa, profile_score         fully connected                 lpa / placed
```

Both models use a **[2, 2, 1]** architecture trained with **Stochastic Gradient Descent (SGD)**.

---

## Dataset

### Regression Dataset
Predict salary (LPA) from CGPA and profile score.

| CGPA | Profile Score | LPA (target) |
|------|--------------|-------------|
| 8 | 8 | 4 |
| 7 | 9 | 5 |
| 6 | 10 | 6 |
| 5 | 12 | 7 |

### Classification Dataset
Predict placement (0 or 1) from CGPA and profile score.

| CGPA | Profile Score | Placed (target) |
|------|--------------|----------------|
| 8 | 8 | 1 |
| 7 | 9 | 1 |
| 6 | 10 | 0 |
| 5 | 5 | 0 |

---

## How It Works

### 1. Forward Pass
Input flows through the network layer by layer.

```python
Z = W.T @ A_prev + b   # Linear transformation
A = sigmoid(Z)          # Activation (classification only)
```

### 2. Loss Calculation

**Regression — Mean Squared Error (MSE):**
```
Loss = (y - ŷ)²
```

**Classification — Binary Cross-Entropy:**
```
Loss = -y·log(ŷ) - (1-y)·log(1-ŷ)
```

### 3. Backward Pass (Backpropagation)
Gradients computed using chain rule, applied layer by layer from output to input.

```
dL/dW1 = dL/dŷ × dŷ/dA1 × dA1/dW1
```

**Weight update rule (Gradient Descent):**
```
W = W + learning_rate × gradient
```

---

## Key Functions

| Function | Purpose |
|----------|---------|
| `initialize_parameters(layer_dims)` | Initialize weights to 0.1, biases to 0 |
| `sigmoid(Z)` | Activation function for classification |
| `linear_forward(A_prev, W, b)` | Single layer forward computation |
| `L_layer_forward(X, parameters)` | Full forward pass across all layers |
| `update_parameters(...)` | Compute gradients and update weights |

---

## Hyperparameters

| Parameter | Regression | Classification |
|-----------|-----------|---------------|
| Learning rate | 0.001 | 0.0001 |
| Epochs | 10 | 10 |
| Weight init | 0.1 (uniform) | 0.1 (uniform) |
| Bias init | 0 | 0 |

---

## Requirements

```bash
pip install numpy pandas
```

---

## Usage

```bash
# Regression
python backpropagation_regression_scratch.py

# Classification
python classification_backpropegation_scratch.py
```

**Expected output:**
```
Epoch - 1  Loss - 12.34
Epoch - 2  Loss - 10.87
...
Epoch - 10 Loss - X.XX
```

---

## Learning Objectives

This project demonstrates:
- Manual implementation of forward propagation
- Backpropagation using chain rule (no autograd)
- Gradient descent weight updates
- Difference between regression vs classification networks
- How sigmoid activation enables binary classification

---

## Author

**Fawad Ahmad**  
BS Artificial Intelligence — University of Haripur  
GitHub: [FawadAhmad-bilal](https://github.com/FawadAhmad-bilal)

*Part of 100 Days of Deep Learning — CampusX curriculum*
