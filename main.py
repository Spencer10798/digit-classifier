import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MNISTDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        image = self.data.iloc[idx, 1:].values.astype(np.float32).reshape(1, 28, 28)
        return torch.tensor(image), torch.tensor(label)

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Add a dropout layer to address train data overfit


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Add a dropout layer to address train data overfit
        x = self.fc2(x)
        return x

def load_data(train_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    return train_data, test_data

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
    train_csv = "archive/mnist_train.csv"
    test_csv = "archive/mnist_test.csv"

    device = torch.device("cuda")

    train_data, test_data = load_data(train_csv, test_csv)

    train_dataset = MNISTDataset(train_data)
    test_dataset = MNISTDataset(test_data)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model( model, train_dataloader, criterion, optimizer, device)
        train_accuracy = evaluate_model(model, train_dataloader, device)
        test_accuracy = evaluate_model(model, test_dataloader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    torch.save(model.state_dict(), 'mnist_classifier.pth')

if __name__ == "__main__":
    main()

