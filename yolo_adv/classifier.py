import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm


class Classifier():
    def __init__(self):
        self.model = self.BinaryResNet50()
        
    def load_dataset(self, root, normalize=True, batch_size=16, shuffle=True):
        # Transformations
        if normalize:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to a common size
                transforms.ToTensor(),          # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to a common size
                transforms.ToTensor(),          # Convert images to PyTorch tensors
            ])
        
        # Load datasets
        dataset = datasets.ImageFolder(root=root, transform=transform)

        # Splitting the dataset into training and validation sets (80-20)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)        

    # Training function
    def train(self, epochs=10, lr=0.001):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in tqdm(range(epochs)):
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
    def evaluate(self, valid_loader=None):
        if valid_loader is None:
            valid_loader = self.valid_loader
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}%')

        
    class BinaryResNet50(nn.Module):
        def __init__(self):
            super(Classifier.BinaryResNet50, self).__init__()
            self.resnet = models.resnet50(pretrained=True)  # Using ResNet50
            for param in self.resnet.parameters():
                param.requires_grad = False  # Freeze the ResNet50 parameters, only final layer will be trained
            
            # Replace the classifier
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2)  # 2 classes: attacked and non-attacked
            )

        def forward(self, x):
            return self.resnet(x)
    