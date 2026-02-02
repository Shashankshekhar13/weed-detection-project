# 🌿 AI-Based Weed Detection Using Image Processing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A deep learning–based image classification system for automated weed detection using EfficientNet-B0 and transfer learning, evaluated on the public DeepWeeds dataset.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Motivation](#-project-motivation)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Evaluation & Visualizations](#-evaluation--visualizations)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [Learning Outcomes](#-learning-outcomes)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

Weeds significantly impact agricultural productivity by competing with crops for nutrients, water, and sunlight. Manual weed identification is labor-intensive and impractical for large-scale farming.

This project explores **deep learning–based image classification** for automated weed detection using **EfficientNet-B0** and **transfer learning** on real-world agricultural images.

> **Academic Disclaimer**  
> This project is developed as a **learning-focused academic exercise**.  
> It is **not** a conference or journal publication and does not claim state-of-the-art performance.

---

## 🌱 Project Motivation

The goal of this project is to gain practical experience with:

- Real-world computer vision datasets
- Transfer learning using pretrained CNNs
- Image preprocessing and data augmentation
- Model evaluation using interpretable metrics
- Error analysis through visual diagnostics

---

## ⭐ Key Features

- Multi-class weed classification (9 classes)
- EfficientNet-B0 pretrained on ImageNet
- Two-stage transfer learning strategy
- Class-wise evaluation using confusion matrix
- Training and validation performance analysis

---

## 📁 Dataset

- **Dataset:** DeepWeeds (Public Dataset)
- **Total Images:** 9,603
- **Classes:** 9  
  - 8 weed species  
  - 1 negative / background class
- **Image Type:** Real-world field images captured under natural conditions

---

## 🧪 Methodology

1. Dataset cleaning and validation  
2. Train / Validation / Test split (60 / 20 / 20)  
3. Image resizing, normalization, and augmentation  
4. Transfer learning using EfficientNet-B0  
5. Two-stage training (feature extraction + fine-tuning)  
6. Quantitative and qualitative evaluation  

---

## 🧠 Model Architecture

- **Base Model:** EfficientNet-B0 (ImageNet pretrained)
- **Custom Head:**
  - Global Average Pooling
  - Dense layer (256 units, ReLU)
  - Softmax output layer (9 classes)

EfficientNet-B0 was selected due to its strong **accuracy–efficiency trade-off**, making it suitable for practical applications.

---

## 🏋️ Training Strategy

### Stage 1 – Feature Extraction
- Backbone frozen
- Classifier head trained
- 15 epochs

### Stage 2 – Fine-Tuning
- Final convolutional layers unfrozen
- Reduced learning rate
- 15 epochs

**Optimizer:** SGD with Momentum  
**Loss Function:** Categorical Cross-Entropy  
**Batch Size:** 32  

---

## 📊 Results

| Metric | Score |
|------|------|
| Test Accuracy | **88.50%** |
| Macro Precision | 88.77% |
| Macro Recall | 88.81% |
| Macro F1-Score | 88.68% |

---

## 📈 Evaluation & Visualizations

### 🔹 Confusion Matrix (Test Set)

Displays class-wise prediction performance and highlights common misclassifications.

![Confusion Matrix](assets/confusion_matrix.png)

---

### 🔹 Training vs Validation Loss (Stage 1 & Stage 2)

Illustrates convergence behavior and the impact of fine-tuning.

![Training vs Validation Loss](assets/stage1_2_loss_combined.png)

---

### 🔹 Training & Validation Accuracy

Shows learning progression during classifier training and fine-tuning stages.

![Accuracy Curves](assets/fig1.png)

> **Note:** ROC-AUC analysis is not included, as confusion matrix and class-wise precision/recall provide more interpretable insights for multi-class image classification tasks.

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- 8 GB RAM recommended
- GPU optional (CPU supported)

### Clone Repository

```bash
git clone https://github.com/Shashankshekhar13/weed-detection-project.git
cd weed-detection-project
