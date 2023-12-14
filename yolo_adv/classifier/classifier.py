import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import signal
from PIL import Image

class Classifier():
    def __init__(self):
        self.model = self.BinaryResNet50()
        self.current_epoch = 0
        self.exp_name = None
        self.optimizer = None
        # Check if CUDA (GPU support) is available
        if torch.cuda.is_available():
            print("GPU is available. Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("GPU is not available. Using CPU.")
        signal.signal(signal.SIGINT, self.signal_handler)
        self.class_names = {0: 'adv', 1: 'real'}

        
    def load_dataset(self, root, normalize=True, batch_size=16, shuffle=True):
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"The specified path does not exist: {root}")

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
    def train(self, exp_name='exp', epochs=10, lr=0.001, valid_period=5, ckpt_period=None):
        self.exp_name = exp_name
        # Check if validation and checkpoint periods are valid
        if valid_period is None:
            valid_period = epochs
        if ckpt_period is None:
            ckpt_period = epochs + 1
            checkpoint_path = None
        else:
            checkpoint_path = f'./yolo_adv/classifier/runs/{self.exp_name}/ckpts'
            os.makedirs(checkpoint_path, exist_ok=True)  # Create the checkpoint directory if it doesn't exist

        self.writer = self.setup_tensorboard_run(f'./yolo_adv/classifier/runs/{self.exp_name}/logging')
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for self.current_epoch in tqdm(range(epochs)):
            with tqdm(self.train_loader, unit="batch", leave=False) as tepoch:
                for i, (images, labels) in enumerate(tepoch):
                    self.optimizer.zero_grad()
                    # Forward
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    # Backward
                    loss.backward()
                    self.optimizer.step()
                    # Log training loss
                    loss_value = loss.item()
                    self.writer.add_scalar('Loss/train', loss_value, self.current_epoch * len(self.train_loader) + i)
                    # Update the progress bar description with the current loss
                    tepoch.set_postfix(loss=loss_value)

            print(f"Epoch [{self.current_epoch}/{epochs}], Loss: {loss_value:.4f}")
            if self.current_epoch > 0 or valid_period == 1:
                if self.current_epoch % valid_period == 0:
                    val_loss = self.validate(criterion)
                    self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
                    print(f"Validation loss: {val_loss:.4f}")
                if self.current_epoch % ckpt_period == 0:
                    torch.save(self.model.state_dict(), f'{checkpoint_path}/{self.current_epoch}.pt')
            
        self.writer.close()
        torch.save(self.model.state_dict(), f'./yolo_adv/classifier/runs/{self.exp_name}/last.pt')
            
    def validate(self, criterion):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

        
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
        
    def load_model(self, model_path):
        """
        Load the model weights from a given path.

        Args:
        model_path (str): Path to the saved model state dictionary.
        map_to_device (bool): If True, map the model to the appropriate device (GPU if available).
        """
        if not hasattr(self, 'model'):
            self.model = self.BinaryResNet50()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict'] 
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(device)  # Move the model to the appropriate device
        
    def predict(self, image_path):
        """
        Perform inference on the input data using the trained model.

        Args:
        input_data (Tensor): The input data on which inference is to be performed.
        model_path (str): Path to the saved model state dictionary.

        Returns:
        The output predictions from the model.
        """
        image = self.process_image(image_path=image_path)        
        self.model.eval()  # Set the model to evaluation mode
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        image = image.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            
        # Convert index to class name
        predicted_class = self.class_names[predicted.item()]

        return predicted_class
    
    # Function to evaluate the model on the dataset
    def evaluate_model_on_dataset(self, dataset_path):
        actual_classes = sorted(os.listdir(dataset_path))
        classes = list(self.class_names.values())
        assert actual_classes == classes, f"Subfolders must be named {classes}, but found {actual_classes}"
        
        correct_predictions = 0
        total_predictions = 0

        for class_name in tqdm(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                with tqdm(os.listdir(class_path), unit="img", leave=False) as tepoch:
                    for img_file in tepoch:
                        img_path = os.path.join(class_path, img_file)
                        predicted_class = self.predict(img_path)
                        if predicted_class == class_name:
                            correct_predictions += 1
                        total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
        
        
        
        
    def process_image(self, image_path):
        # Define the same transformations as used during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Assuming you used this size during training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Transform the image
        transformed_image = transform(image)

        # Add an extra batch dimension, since PyTorch treats all inputs as batches
        image_batch = transformed_image.unsqueeze(0)

        return image_batch
        
    def setup_tensorboard_run(self, run_name):
        log_dir = os.path.join('runs', run_name)
        if os.path.exists(log_dir):
            user_input = input(f"The run '{run_name}' already exists. Overwrite? (y/n): ")
            if user_input.lower() == 'y':
                # Overwrite the existing run directory
                shutil.rmtree(log_dir)
            else:
                print("Training cancelled.")
                exit()
        writer = SummaryWriter(log_dir)
        return writer

    # Signal handler function
    def save_checkpoint_on_interrupt(self, filename='last.pt'):
        print("\nCtrl+C pressed. Saving checkpoint before exiting.")
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f'./yolo_adv/classifier/runs/{self.exp_name}/{filename}')
        
    def signal_handler(self, signal_received, frame):
        self.save_checkpoint_on_interrupt()
        print("Checkpoint saved. Exiting.")
        exit(0)
        
    class BinaryResNet50(nn.Module):
        def __init__(self):
            super(Classifier.BinaryResNet50, self).__init__()
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Using ResNet50
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
    