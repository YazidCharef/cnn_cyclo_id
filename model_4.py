import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Model definition
class StormCNET1(nn.Module):
    def __init__(self, input_channels):
        super(StormCNET1, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 2
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 3
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
        )
        
        # Calculate the output size of CNN layers
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 241, 321)
            sample_output = self.cnn_layers(sample_input)
            self.flat_features = sample_output.view(1, -1).size(1)
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.SiLU(),
            nn.Linear(64, 4),  # 4 classes: cyclogenesis, full typhoon, cyclolysis, no cyclone
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def load_data(info):
    print("Loading data...")
    all_data_x = []
    all_data_y = []
    for file in info['chunk_files']:
        with np.load(os.path.join(info['output_dir'], file)) as data:
            all_data_x.append(torch.from_numpy(data['X']).float())
            all_data_y.append(torch.from_numpy(data['y']).long())
    
    X = torch.cat(all_data_x, dim=0)
    y = torch.cat(all_data_y, dim=0)
    
    dataset = TensorDataset(X, y)
    train_size = info['train_size']
    test_size = info['test_size']
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if batch_idx % 100 == 0:
            print(f'  Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    return test_loss, accuracy, all_preds, all_labels

def calculate_f1_scores(y_true, y_pred):
    class_names = ['cyclogenesis', 'full typhoon', 'cyclolysis', 'no cyclone']
    f1_scores = {}
    
    for i, class_name in enumerate(class_names):
        true_positives = np.sum((np.array(y_true) == i) & (np.array(y_pred) == i))
        false_positives = np.sum((np.array(y_true) != i) & (np.array(y_pred) == i))
        false_negatives = np.sum((np.array(y_true) == i) & (np.array(y_pred) != i))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[class_name] = f1
    
    # Calculate global F1-score (average of F1-scores per class)
    f1_scores['global'] = np.mean(list(f1_scores.values()))
    
    return f1_scores

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data information...")
    info = torch.load('cyclone_data_prepared_info.pt')
    train_loader, test_loader = load_data(info)

    model = StormCNET1(input_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    print("Starting training...")
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        print(f'Epoch time: {epoch_time:.2f}s | Total time: {total_time:.2f}s')

    print(f"Training completed. Total time: {time.time() - start_time:.2f}s")

    # Final evaluation and metric calculation
    print("Final model evaluation...")
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    f1_scores = calculate_f1_scores(all_labels, all_preds)

    print("\nF1-scores by category:")
    for class_name, f1 in f1_scores.items():
        if class_name != 'global':
            print(f"  {class_name}: {f1:.4f}")
    print(f"\nGlobal F1-score: {f1_scores['global']:.4f}")

    # Calculate and display confusion matrix
    cm = np.zeros((4, 4), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1
    print("\nConfusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, ['cyclogenesis', 'full typhoon', 'cyclolysis', 'no cyclone'])

    # Save the model
    torch.save(model.state_dict(), 'storm_cnet1_model.pth')
    print("Model saved as 'storm_cnet1_model.pth'")

    # Learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Learning curves saved as 'learning_curves.png'")

if __name__ == "__main__":
    main()