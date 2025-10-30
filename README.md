# Restricted Boltzmann Classifier

A clean, minimal implementation of a **Restricted Boltzmann Machine (RBM)** for classification and basic image reconstruction, trained on MNIST.
Originally built for experimentation, this version has been rewritten and modularized to be readable, hackable, and reproducible.

---

## ⚡ Overview

This repository demonstrates how a shallow probabilistic model can **learn digit representations** and perform simple classification using a contrastive divergence–trained RBM.

The project contains:

-   **Custom RBM implementation** (`model/rbm.py`)
-   **MNIST utilities and visualization helpers** (`model/utils.py`)
-   **Command-line interface for training and evaluation** (`main.py`)
-   **Full experimental notebook** in `/notebooks/`

No TensorFlow. No PyTorch. Just raw NumPy and scikit-learn.

---

## 🧩 Project Structure

```
rbm_project/
│
├── main.py # Entry point for CLI-based training/testing
├── requirements.txt
│
├── model/
│ ├── rbm.py # RBM implementation (Contrastive Divergence)
│ └── utils.py # Dataset loading + visualization helpers
│
├── notebooks/
│ └── restricted_boltzmann_classifier.ipynb
│
└── README.md
```

---

## 🚀 Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/restricted-boltzmann-classifier.git
cd restricted-boltzmann-classifier
pip install -r requirements.txt
```

---

## 🧪 Usage

You can run the RBM training and evaluation directly from the CLI:

```
python main.py --n-samples 10000 --n-hidden 128 --n-epochs 10 --learning-rate 0.1
```

**Arguments:**

| Flag              | Description                    | Default |
| ----------------- | ------------------------------ | ------- |
| `--n-samples`     | Number of MNIST samples to use | 10000   |
| `--n-hidden`      | Hidden layer size              | 128     |
| `--n-epochs`      | Training epochs                | 10      |
| `--learning-rate` | Learning rate for CD updates   | 0.1     |
| `--batch-size`    | Batch size                     | 64      |
| `--binarize`      | Use binary visible units       | `False` |

Example:

```
python main.py --n-samples 5000 --n-hidden 256 --n-epochs 15
```

---

## 📈 Results

RBM achieves decent (80%) classification accuracy after two training passes on MNIST under the current configuration.
Earlier experimental versions reached ~90%, suggesting potential gains with tuned hyperparameters or checkpointed training.

Reconstruction quality is visibly coarse but representative, ideal for understanding how low-dimensional binary encodings emerge.

---

## 🧰 Modding Tips

To improve results, consider:

-   Increasing `n_hidden`
-   Lowering the learning rate
-   Implementing **Persistent Contrastive Divergence (PCD)**
-   Adding **checkpointing** or **momentum** terms

To visualize learned weights, import your trained RBM and call:

```
rbm.visualize_weights()
```

(if you implement it yourself — not included by default)

---

## 📒 Notebook

The "/notebooks/restricted_boltzmann_classifier.ipynb" file contains:

-   Step-by-step walkthrough
-   Training results and plots
-   Reconstruction visuals

Everything runs standalone, no need to retrain unless you want fresh results.

---

## 🧠 Background

A Restricted Boltzmann Machine (RBM) is a stochastic neural network that learns a **joint probability distribution** between visible and hidden units.  
It forms the basis for **Deep Belief Networks** and pre-deep-learning unsupervised feature learning techniques.

Training uses **Contrastive Divergence (CD-k)**, where the model learns to approximate the data distribution through Gibbs sampling.

---

## 📦 Requirements

```
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.2
pandas==2.2.3
tqdm==4.66.5  # optional
```

---

## ⚖️ License

MIT License — do whatever you want, but maybe give credit if you want.
