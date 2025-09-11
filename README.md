AI-Based Weed Detection Using Image Processing

A deep learning project to classify agricultural weeds using the EfficientNet-B0 architecture and transfer learning.
This system achieves 88.50% test accuracy on the public DeepWeeds dataset, demonstrating an effective approach for automated weed identification in precision agriculture.

📌 Project Summary

Objective:
Build an accurate and efficient AI model for multi-class weed classification.

Model:
EfficientNet-B0 (pre-trained on ImageNet)

Dataset:
DeepWeeds – 9,603 images across 9 weed species.

Methodology:

Stage 1 (15 Epochs): Trained classifier head with frozen base model

Stage 2 (15 Epochs): Fine-tuned final layers with reduced learning rate

Key Result:
✅ Achieved 88.50% accuracy and 88.68% macro F1-score on the held-out test set.

🛠 Tech Stack

Language: Python

Frameworks: PyTorch, Torchvision

Libraries: scikit-learn, NumPy, Matplotlib

🚀 How to Run
1️⃣ Setup

Clone the repository and install dependencies.
A virtual environment is recommended.

git clone https://github.com/Shashankshekhar13/weed-detection-project.git
cd weed-detection-project
pip install -r requirements.txt


Make sure requirements.txt contains PyTorch, Torchvision, scikit-learn, NumPy, Matplotlib, etc.

2️⃣ Training

To train the model from scratch (30 epochs total):

python scripts/train.py


This script will save the final model weights as efficientnetb0_final.pth.

3️⃣ Evaluation

To evaluate the final model on the test set:

python scripts/test.py


This will:

Print a detailed classification report

Generate a confusion matrix plot (fig3_confusion_matrix.png)

📊 Performance Highlights
Metric	Score
Test Accuracy	88.50%
Macro Precision	88.77%
Macro Recall	88.81%
Macro F1-Score	88.68%
📷 Confusion Matrix (Test Set)
