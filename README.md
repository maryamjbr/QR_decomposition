# Handwritten Character Classification using QR Decomposition

This repository implements and evaluates a **QR-decomposition–based approach** for handwritten character classification using the **EMNIST Letters dataset**.  
The project combines classical linear algebra (QR decomposition, Householder reflections, Givens rotations) with subspace-based classification methods, and explores efficient **QR updating algorithms** when new data is added.

---

## 📌 Overview
Handwritten character recognition is a fundamental challenge in pattern recognition and machine learning.  
In this project, I leverage **QR decomposition** to build subspaces for each letter class (A–Z), and classify new handwritten samples using the **nearest-subspace method**.  

Two main parts of the project:
1. **Initial Classification**  
   - Construct training matrices for each class  
   - Perform QR decomposition (Householder reflections)  
   - Classify test samples via least squares residual minimization  
   - Evaluate per-class and overall accuracy  

2. **QR Updating with Givens Rotations**  
   - Efficiently update QR decomposition when new training data is added  
   - Avoid recomputing the full QR decomposition  
   - Analyze classification performance before and after updating  

---

## 📂 Dataset
We use the **EMNIST Letters** dataset:
- **26 balanced classes (A–Z)**  
- Each image: **28 × 28 grayscale** → flattened to **784-dimensional vectors**  
- For each class:  
  - 200 samples used for training  
  - 20 samples used for testing  

Dataset reference: [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## ⚙️ Methods

### 1. QR Decomposition for Classification
- Each class `i` → training matrix `A_i` (size 784×200)  
- Apply **Householder reflections** to compute `A_i = Q_i R_i`  
- For a test vector `z`, solve least-squares problem:  

  \[
  x = \arg\min_x \|z - A_i x\|_2
  \]

- Compute residuals:

  \[
  r_i = \|z - A_i x\|_2
  \]

- Predicted class = subspace with smallest residual.

### 2. QR Updating with Givens Rotations
- Instead of recomputing QR when new samples are added, use **Givens rotations** for efficient updates.  
- Functions implemented:
  - `givens(a, b)` → stable computation of cosine/sine parameters  
  - `update_qr_with_new_column(A, Q, R, x)` → incremental update when adding a new column  
- This method reduces computational cost and preserves numerical stability.

---

## 📊 Results

- **Initial Training (200 samples/class)**:  
  - Overall accuracy ≈ **70%** on EMNIST Letters test set  
  - Some letters (K, V, Z) show high accuracy (>90%), others (L, D, Q) are more challenging  

- **After Adding New Data (QR Updating with Givens)**:  
  - Accuracy slightly decreased to ≈ **68%**  
  - Improvements in some classes (e.g., N, M, H, Q), declines in others (e.g., G, L)  
  - Trade-off observed due to added noise and structural overlap between classes  

- **Confusion Matrix Analysis**:  
  - High confusion between visually similar letters (e.g., I–L, G–Q–C, D–P)  
  - Stable performance on distinct classes (e.g., M, N, V, Z)  



## 📁 Repository Structure
