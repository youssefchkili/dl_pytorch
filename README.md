# Cybersecurity Intrusion Detection with PyTorch 🔒🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project implementing a binary classification neural network to detect network intrusion attacks using PyTorch. This project demonstrates end-to-end machine learning workflows including data preprocessing, custom dataset creation, model training with learning rate scheduling, and real-time inference.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project builds a neural network to classify network sessions as either **Normal (0)** or **Attack (1)** based on network traffic features. The model achieves robust performance using binary cross-entropy loss with Adam optimization and learning rate scheduling.

**Key Features:**
- Custom PyTorch Dataset and DataLoader implementation
- Mixed data type handling (numerical + categorical)
- Learning rate scheduling with ReduceLROnPlateau
- GPU acceleration support
- Real-time inference pipeline with preprocessing
- Comprehensive evaluation using AUC-ROC metric

## 📊 Dataset

**Source:** [Kaggle - Cybersecurity Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)

**Characteristics:**
- **Size:** 9,537 network sessions
- **Features:** 9 total (6 numerical, 3 categorical)
- **Target:** Binary classification (`attack_detected`: 0=Normal, 1=Attack)
- **Balance:** Approximately 60% Normal / 40% Attack

**Features:**
| Feature | Type | Description |
|---------|------|-------------|
| `network_packet_size` | Numerical | Size of data packets transmitted |
| `protocol_type` | Categorical | Network protocol (TCP/UDP/ICMP) |
| `encryption_used` | Categorical | Encryption standard (TLS/AES/None) |
| `login_attempts` | Numerical | Number of login tries |
| `session_duration` | Numerical | Length of connection (seconds) |
| `failed_logins` | Numerical | Failed authentication attempts |
| `unusual_time_access` | Binary | Access during off-hours (0/1) |
| `ip_reputation_score` | Numerical | Risk score of source IP (0-1) |
| `browser_type` | Categorical | Client browser (Chrome/Firefox/Safari/Unknown) |

**Preprocessing Notes:**
- Features are pre-normalized in the dataset (mean≈0, std≈1)
- Categorical variables encoded using Label Encoding
- Session ID columns removed (non-predictive)
- Train/Test split: 80/20 with stratification

## ⚙️ Installation

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
scikit-learn
pandas
numpy
matplotlib
seaborn
