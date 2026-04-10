# Power Side-Channel Attack on AES + ML Key Recovery

## Overview

This project demonstrates a **side-channel attack on AES encryption** using both:

* **Classical Cryptanalysis** → Correlation Power Analysis (CPA)
* **Machine Learning** → Multi-Layer Perceptron (MLP)

Instead of attacking AES mathematically, this project exploits **power consumption leakage** to recover the secret key.


##  Objectives

* Simulate realistic **power traces** using leakage models
* Implement **Correlation Power Analysis (CPA)**
* Apply **Machine Learning for cryptanalysis**
* Compare classical vs ML-based key recovery


##  Key Concepts

###  Side-Channel Attacks

Attacks that use **physical information leakage** (power, timing, EM signals) instead of algorithmic weaknesses.

### Hamming Weight Model

Power consumption is modeled as:

HW(SBOX(plaintext ⊕ key))

Where HW = number of 1s in binary representation.

### Correlation Power Analysis (CPA)

* Try all 256 possible key values
* Compute correlation between predicted and actual power
* Correct key → highest correlation

###  Machine Learning Attack

* Train neural network on power traces
* Predict leakage patterns automatically
* Recover key via probability scoring


##  Features

*  Simulated noisy power traces
*  AES S-Box leakage modeling
*  Full CPA implementation
*  Neural network-based key recovery (PyTorch)
*  Visualization of attack results
*  Comparison: Classical vs ML approach


##  Methodology

### 1. Trace Simulation

* Random plaintexts generated
* Leakage inserted at a fixed time index
* Gaussian noise added

### 2. CPA Attack

* Build Hamming Weight model
* Compute Pearson correlation
* Rank key candidates

### 3. ML Attack

* Input: plaintext + power trace
* Output: Hamming Weight class (0–8)
* Evaluate all key hypotheses


##  Output

The program generates:

*  Correlation vs time plots
*  Key ranking graphs
*  ML score distributions
*  Sample power traces

##  How to Run

### 1. Install dependencies

```bash
pip install numpy matplotlib torch scikit-learn
```

### 2. Run the project

```bash
python main.py
```

---

## 🧪 Default Parameters

* Trace Length: 50
* Leakage Point: 10
* Noise Std: 1.5
* Training Traces: 2000
* Test Traces: 500

---

##  Security Insight

This project demonstrates:

> Even strong encryption like AES can be broken
> if implementation leaks physical information.

### Countermeasures:

* Masking
* Noise injection
* Constant-time implementations



##  Relevance

* Side-channel security research
* Hardware cryptography
* Machine learning in cybersecurity

##  Tech Stack

* Python
* NumPy
* Matplotlib
* PyTorch

##  Future Work

* Use real datasets (ASCAD)
* Deep learning (CNN-based attacks)
* Multi-byte key recovery
* Hardware-level experiments


##  Author

Manya Kumar

##  Summary

This project bridges:

* **Cryptography** 
* **Hardware Security** 
* **Machine Learning** 

to demonstrate a real-world vulnerability in secure systems.
