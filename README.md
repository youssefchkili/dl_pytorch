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
Setup
bash
Copy
# Clone the repository
git clone https://github.com/yourusername/cybersecurity-intrusion-detection.git
cd cybersecurity-intrusion-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in /data folder
🗂️ Project Structure
Copy
cybersecurity-intrusion-detection/
├── data/
│   └── cybersecurity_intrusion_data.csv
├── notebooks/
│   └── intrusion_detection.ipynb          # Main Colab/Notebook
├── src/
│   ├── model.py                           # Neural network architecture
│   ├── dataset.py                         # Custom PyTorch Dataset
│   ├── train.py                           # Training loop
│   └── predict.py                         # Inference pipeline
├── models/
│   └── best_intrusion_model.pth           # Saved model weights
├── utils/
│   ├── scaler.pkl                         # Fitted StandardScaler
│   └── label_encoders.pkl                 # Categorical encoders
├── README.md
└── requirements.txt
🚀 Usage
1. Training the Model
Python
Copy
from src.train import train_model

# Initialize model
model = IntrusionDetector(input_size=9, hidden_size=128)

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
Training Configuration:
Optimizer: Adam (lr=0.001)
Loss: Binary Cross Entropy (BCELoss)
Scheduler: ReduceLROnPlateau (patience=5, factor=0.1)
Batch Size: 64
Epochs: 50 (with early stopping capability)
2. Making Predictions
Python
Copy
from src.predict import predict_intrusion

# Example: Normal network session
prediction, confidence = predict_intrusion(
    network_packet_size=0.05,      # Normalized value
    protocol_type='TCP',
    encryption_used='TLS',
    login_attempts=0.0,            # Low attempts
    session_duration=0.36,         # Average duration
    failed_logins=0.0,             # No failures
    unusual_time_access=-1.5,      # Business hours
    ip_reputation_score=-0.7,      # Trusted IP
    browser_type='Chrome'
)

print(f"Prediction: {'Attack' if prediction == 1 else 'Normal'}")
print(f"Confidence: {confidence:.2%}")
Input Format Note:
The dataset comes pre-normalized. For real-world deployment, inputs should be standardized using the same scaler fitted on training data, or use normalized values relative to training distributions.
🧠 Model Architecture
Python
Copy
IntrusionDetector(
  (layer1): Linear(in_features=9, out_features=128)
  (bn1): BatchNorm1d(128)
  (relu): ReLU()
  (dropout): Dropout(p=0.3)
  (layer2): Linear(in_features=128, out_features=64)
  (bn2): BatchNorm1d(64)
  (output): Linear(in_features=64, out_features=1)
  (sigmoid): Sigmoid()
)
Design Choices:
2 Hidden Layers (128→64): Sufficient capacity for tabular data without overfitting
Batch Normalization: Stabilizes training and reduces internal covariate shift
Dropout (0.3): Prevents overfitting on this medium-sized dataset
ReLU Activation: Non-linearity for complex pattern recognition
Output Sigmoid: Compresses output to probability (0-1) for binary classification
Why Adam + ReduceLROnPlateau?
Adam: Handles sparse gradients well (categorical features) and adapts learning rates per parameter
Scheduler: Automatically reduces learning rate when validation loss plateaus, allowing precise convergence to the loss minimum without manual tuning
📈 Results
Performance Metrics
Table
Copy
Metric	Score
Validation AUC	~0.89-0.92
Validation Loss	~0.25-0.30
Training Loss	~0.20-0.25
Learning Curves
Include plots of training/validation loss over epochs showing convergence
Feature Importance
Include permutation importance plot showing:
failed_logins - Most predictive of attacks
ip_reputation_score - Critical for identifying malicious sources
login_attempts - High volume indicates brute force
network_packet_size - Anomalous sizes indicate data exfiltration
🎓 Key Learnings & Challenges Solved
1. Data Type Handling in PyTorch
Challenge: RuntimeError: mat1 and mat2 must have the same dtype
Solution: Explicitly convert numpy arrays to float32 before tensor creation:
Python
Copy
X_train = X_train.astype(np.float32)  # Prevents default float64
2. BCELoss Shape Requirements
Challenge: ValueError: Target size must match input size
Solution: Squeeze model outputs or unsqueeze labels to match dimensions [batch_size] vs [batch_size, 1].
3. Pre-normalized Dataset Characteristics
Insight: The Kaggle dataset comes with pre-standardized features (mean≈0, std≈1).
Implication: Raw real-world values (e.g., session_duration=1800s) must be normalized using training statistics before inference, or the model sees extreme outliers (scaled value=2291) and predicts Attack with 100% confidence.
4. Label Encoder Persistence
Challenge: AttributeError: 'builtin_function_or_method' object has no attribute 'transform'
Root Cause: Accidentally stored len instead of encoder object le.
Lesson: Always verify object types in preprocessing pipelines.
5. Feature Order Consistency
Critical: Inference features must match training column order exactly. Mismatch causes the model to interpret session_duration as network_packet_size, leading to nonsensical predictions.
🔮 Future Improvements
Class Imbalance Handling: Implement weighted BCELoss or SMOTE oversampling if dataset is imbalanced
Architecture Upgrades:
Try TabNet for tabular data
Implement Attention mechanisms
Experiment with deeper networks (3-4 layers)
Threshold Tuning: Optimize classification threshold using Precision-Recall curve rather than default 0.5
Feature Engineering: Create interaction features (login_attempts × failed_logins)
Anomaly Detection: Implement Autoencoder approach for unsupervised anomaly detection on unlabeled network traffic
Deployment: Convert to ONNX format for edge deployment on network appliances
🛡️ Ethical Considerations & Limitations
False Positives: High false positive rates in intrusion detection can lead to alert fatigue for security teams
Adversarial Attacks: Neural networks can be fooled by carefully crafted network packets; consider adversarial training
Bias: Model performance may vary across different network environments; retraining recommended for production deployment
Privacy: Ensure compliance with data privacy regulations when collecting network traffic for training
📚 References
Dataset: Cybersecurity Intrusion Detection Dataset - Kaggle
PyTorch Documentation: torch.nn.Module
Understanding AUC-ROC: Google Developers - Classification
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Your Name - GitHub - LinkedIn
Project completed as part of deep learning specialization with PyTorch.
⭐ Star this repository if you found it helpful!
