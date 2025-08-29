# scripts/test.py

import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Absolute import from scripts folder
from preprocessing import load_datasets

def load_model(num_classes, device, model_path="efficientnetb0.pth"):
    """Instantiate EfficientNet-B0, swap out the head, load weights."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, loader, device):
    """Run inference on loader, return accuracy plus lists of true/pred."""
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return 100 * correct / total, all_labels, all_preds

if __name__ == "__main__":
    # Paths (relative to project root)
    data_dir   = "AIML/dataset_cleaned"
    model_path = "efficientnetb0.pth"
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")

    # Load data
    _, _, test_loader, class_names = load_datasets(data_dir, batch_size=batch_size)

    # Load model
    model = load_model(num_classes=len(class_names), device=device, model_path=model_path)

    # Evaluate
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\n🧪 Test Accuracy: {test_acc:.2f}%\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
