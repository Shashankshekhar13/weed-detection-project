import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from preprocessing import load_datasets  # ✅ Import from your custom script

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        val_acc = evaluate_model(model, val_loader, device)
        print(f"✅ Epoch {epoch + 1} completed | Train Accuracy: {100. * correct / total:.2f}% | Val Accuracy: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), "efficientnetb0.pth")
    print("🎉 Model saved to efficientnetb0.pth")


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    data_dir = "AIML/dataset_cleaned"
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")

    # Load datasets
    train_loader, val_loader, test_loader, class_names = load_datasets(data_dir, batch_size=batch_size)

    # Load pretrained EfficientNet-B0
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Modify classifier to match number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

