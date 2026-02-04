# fed-diffusion-image-classification

This project explores a **diffusion-model-based data augmentation framework for federated image classification** under Non-IID data distributions. By training diffusion models in a federated manner and leveraging generated samples to balance class distributions, the framework significantly improves global classification performance while preserving data privacy.

📄 **Paper**: *Data Augmentation for Federated Image Classification Using Diffusion Models*  
🔗 **Code**: https://github.com/VimGeoTeam/fed-diffusion-image-classification  

---

## 📌 Overview

Federated Learning (FL) enables collaborative model training without sharing raw data, making it suitable for privacy-sensitive applications. However, real-world federated data is often **Non-IID**, resulting in:

- Label shift  
- Severe class imbalance  
- Slow convergence and poor generalization  

To address these challenges, this project introduces a **federated diffusion-based data augmentation framework** that:

1. Trains diffusion models locally on each client.
2. Aggregates diffusion models on the server to form a global generative model.
3. Uses the global diffusion model to generate high-quality synthetic samples.
4. Distributes generated samples to balance local class distributions.
5. Trains a federated classification model (ResNet-18) using both real and generated data.

---

## 🧠 Key Contributions

- ✅ Proposes a **federated diffusion generation framework** for privacy-preserving cross-client data augmentation.
- ✅ Designs a **class-imbalance-aware augmentation strategy** based on missing-class ratios.
- ✅ Demonstrates significant improvements on CIFAR-10 and CIFAR-100 under severe Non-IID conditions (Dirichlet α=0.1).

---

## 📊 Experimental Results

Under Non-IID data distribution with Dirichlet coefficient α = 0.1:

| Dataset   | Baseline Accuracy | With Diffusion Augmentation |
|-----------|------------------|-----------------------------|
| CIFAR-10  | 46.76%           | **54.64%**                  |
| CIFAR-100 | 21.31%           | **25.57%**                  |

The fine-tuned global diffusion model generates samples closer to the real data distribution, effectively balancing local datasets and improving global classification accuracy.

---

## 🏗️ Project Structure and File Descriptions

```bash
fed-diffusion-image-classification/
├── ImageNet_Diffusion_model_center.ipynb
├── Cifar10_Diffusion_model_non_IID.ipynb
├── Cifar100_Diffusion_model_non_IID.ipynb
├── Cifar10_Resnet18_model_non_IID_noise_diffusion_compare.ipynb
├── Cifar100_Resnet18_model_non_IID_noise_diffusion_compare.ipynb
├── requirements.txt
└── README.md

## 📂 File Descriptions and Relation to the Paper

### 🔹 Diffusion Model Training (Generative Stage)

#### 1️⃣ `ImageNet_Diffusion_model_center.ipynb`

**Purpose:**  
Implements centralized (non-federated) diffusion model pretraining on ImageNet.

**Relation to Paper:**  
Corresponds to the global diffusion model pretraining stage, validating the base generative capability of the denoising diffusion probabilistic model (DDPM) before federated deployment.

---

#### 2️⃣ `Cifar10_Diffusion_model_non_IID.ipynb`

**Purpose:**  
Implements federated training of diffusion models under Non-IID CIFAR-10 distributions.

**Relation to Paper:**  
Directly implements the paper’s federated diffusion generation framework, where each client trains a local diffusion model and uploads parameters for server-side aggregation to form a global generative model.

---

#### 3️⃣ `Cifar100_Diffusion_model_non_IID.ipynb`

**Purpose:**  
Same as above, but on CIFAR-100 with more severe class imbalance and higher category complexity.

**Relation to Paper:**  
Supports experimental validation of the framework under more challenging Non-IID conditions, corresponding to the CIFAR-100 experiments reported in the paper.

---

### 🔹 Federated Classification (Discriminative Stage)

#### 4️⃣ `Cifar10_Resnet18_model_non_IID_noise_diffusion_compare.ipynb`

**Purpose:**  
Trains federated ResNet-18 classifiers on CIFAR-10 using:
- only real data (baseline),
- real + noise-augmented data,
- real + diffusion-generated data.

**Relation to Paper:**  
Implements the comparative evaluation framework used in the paper to demonstrate that diffusion-based augmentation outperforms traditional noise-based methods under Non-IID conditions.

---

#### 5️⃣ `Cifar100_Resnet18_model_non_IID_noise_diffusion_compare.ipynb`

**Purpose:**  
Same as above, but on CIFAR-100.

**Relation to Paper:**  
Corresponds to the CIFAR-100 classification experiments validating the generalizability of the proposed data augmentation strategy across datasets.

---

### 🔹 Environment Setup

#### 6️⃣ `requirements.txt`

**Purpose:**  
Lists all Python dependencies required to reproduce the experiments, including PyTorch, torchvision, numpy, and federated learning utilities.

**Relation to Paper:**  
Ensures reproducibility and consistency with the experimental settings reported in the paper.

---

## 📜 Citation

If you find this work useful, please cite:

> Huang, J., Wu, M., Wang, S., Lai, Y., & Yu, R.  
> *Data Augmentation for Federated Image Classification Using Diffusion Models*.  
> Journal of Computer Research and Development, 2025.  
> DOI: 10.19678/j.issn.1000-3428.00253368
