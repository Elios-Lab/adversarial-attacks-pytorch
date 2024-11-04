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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Classifier():
    def __init__(self):
        self.model = self.ResNet50()
        self.current_epoch = 0
        self.exp_name = None
        self.optimizer = None
        # Check if CUDA (GPU support) is available
        if torch.cuda.is_available():
            print("GPU is available. Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        elif torch.backends.mps.is_available():
            print('MPS is available. Using MPS.')
        else:
            print("GPU is not available. Using CPU.")
        signal.signal(signal.SIGINT, self.signal_handler)
        self.class_names = {0: 'clean', 1: 'pixle', 2: 'poltergeist'}

        
    def load_dataset(self, root, normalize=True, batch_size=16, shuffle=True):       
        if not os.path.exists(root):
            raise FileNotFoundError(f"The specified path does not exist: {root}")

        # Transformations
        if normalize:
            transform = transforms.Compose([
                # transforms.Resize((224, 224)),  # Resize images to a common size
                transforms.ToTensor(),          # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
            ])
        else:
            transform = transforms.Compose([
                # transforms.Resize((224, 224)),  # Resize images to a common size
                transforms.ToTensor(),          # Convert images to PyTorch tensors
            ])
        
        # Load datasets from separate train, validation, and test folders
        train_path = os.path.join(root, 'train')
        val_path = os.path.join(root, 'val')
        test_path = os.path.join(root, 'test')

        if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"One or more dataset folders (train, val, test) do not exist in the specified path: {root}")

        train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        # self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
     

    # Training function
    def train(self, exp_name='exp', epochs=10, lr=0.001, valid_period=5, ckpt_period=None, patience=None):
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

        self.writer = self.setup_tensorboard_run(f'./yolo_adv/classifier/runs/{self.exp_name}/tb_logs')
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.model.train()
        
        # Early stopping criteria
        best_val_loss = float('inf')
        patience_counter = 0
        
        for self.current_epoch in tqdm(range(epochs)):
            self.model.train()
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
                    val_loss, accuracy, precision, recall, f1 = self.validate(criterion)
                    self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
                    self.writer.add_scalar('Accuracy/val', accuracy, self.current_epoch)
                    self.writer.add_scalar('Precision/val', precision, self.current_epoch)
                    self.writer.add_scalar('Recall/val', recall, self.current_epoch)
                    self.writer.add_scalar('F1/val', f1, self.current_epoch)
                    print(f"Validation loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            print("Early stopping triggered")
                            break
                if self.current_epoch % ckpt_period == 0:
                    torch.save(self.model.state_dict(), f'{checkpoint_path}/{self.current_epoch}.pt')
            
        self.writer.close()
        torch.save(self.model.state_dict(), f'./yolo_adv/classifier/runs/{self.exp_name}/last.pt')
            
    def validate(self, criterion):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in self.valid_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())

                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return avg_loss, accuracy, precision, recall, f1


        
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
            self.model = self.ResNet50()

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict'] 
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(device)  # Move the model to the appropriate device
        
    def predict(self, image_path=None, image:Image=None):
        """
        Perform inference on the input data using the trained model.

        Args:
        input_data (Tensor): The input data on which inference is to be performed.
        model_path (str): Path to the saved model state dictionary.

        Returns:
        The output predictions from the model.
        """
        image = self.process_image(image_path=image_path, image=image)      
        self.model.eval()  # Set the model to evaluation mode
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
        actual_classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        classes = list(self.class_names.values())
        assert actual_classes == classes, f"Subfolders must be named {classes}, but found {actual_classes}"
        
        #correct_predictions = 0
        #total_predictions = 0

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        all_labels = []
        all_predictions = []

        for class_name in tqdm(sorted(os.listdir(dataset_path)), total=2):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                with tqdm(os.listdir(class_path), unit="img", leave=False, initial=1) as tepoch:
                    for img_file in tepoch:
                        img_path = os.path.join(class_path, img_file)
                        predicted_class = self.predict(img_path)
                        all_labels.append(class_name)
                        all_predictions.append(predicted_class)
                        if predicted_class == class_name == 'pixle' or predicted_class == class_name == 'poltergeist':
                            true_positives += 1
                        elif predicted_class == class_name == 'clean':
                            true_negatives += 1
                        elif (predicted_class == 'pixle' or predicted_class == 'poltergeist') and class_name == 'clean':
                            false_positives += 1
                        elif predicted_class == 'clean' and (class_name == 'pixle' or class_name == 'poltergeist'):
                            false_negatives += 1
                        else:
                            print(f"Unexpected class: {predicted_class} for image classified as {class_name}")
                            
                        # elif predicted_class == 'adv' and class_name == 'real':
                        #     false_positives += 1
                        # elif predicted_class == 'real' and class_name == 'adv':
                        #     false_negatives += 1
                        # else:
                        #     true_negatives += 1
                            # if predicted_class not in os.listdir(dataset_path):
                            #     false_negatives += 1
                        # if predicted_class == class_name:
                        #     correct_predictions += 1
                        # total_predictions += 1
        # Calculate metrics using sklearn
        s_accuracy = accuracy_score(all_labels, all_predictions)
        s_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        s_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        s_f1_score = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        print('Metrics in one-vs-all setting:')
        print(f"Accuracy: {s_accuracy}, Precision: {s_precision}, Recall: {s_recall}, F1 Score: {s_f1_score}")
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if true_positives + true_negatives + false_positives + false_negatives > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        TPR = true_positives / (true_positives + false_negatives + true_negatives + false_positives)
        FPR = false_positives / (true_positives + false_negatives + true_negatives + false_positives)
        TNR = true_negatives / (true_positives + false_negatives + true_negatives + false_positives)
        FNR = false_negatives / (true_positives + false_negatives + true_negatives + false_positives)

        # Plot the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=classes)
        print('Metrics in clean-vs-all setting:')
        print(f"True Positives: {true_positives}, True Negatives: {true_negatives}, False Positives: {false_positives}, False Negatives: {false_negatives}")
        #accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy, precision, recall, f1, TPR, FPR, TNR, FNR, cm
    
    def plot_cm(self, cm):
        classes = list(self.class_names.values())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

        
    def process_image(self, image_path=None, image:Image=None):
        # Define the same transformations as used during training
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # Assuming you used this size during training
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the image
        if image is not None:
            image = image.convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        # Transform the image
        transformed_image = transform(image)

        # Add an extra batch dimension, since PyTorch treats all inputs as batches
        image_batch = transformed_image.unsqueeze(0)

        return image_batch
        
    def setup_tensorboard_run(self, run_path):
        # log_dir = os.path.join('runs', run_path)
        if os.path.exists(run_path):
            user_input = input(f"The run '{run_path}' already exists. Overwrite? (y/n): ")
            if user_input.lower() == 'y':
                # Overwrite the existing run directory
                shutil.rmtree(run_path)
            else:
                print("Training cancelled.")
                exit()
        writer = SummaryWriter(run_path)
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
        
    class ResNet50(nn.Module):
        def __init__(self):
            super(Classifier.ResNet50, self).__init__()
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Using ResNet50
            for param in self.resnet.parameters():
                param.requires_grad = False  # Freeze the ResNet50 parameters, only final layer will be trained
            
            # Replace the classifier
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 3)  # 2 classes: attacked and non-attacked
            )

        def forward(self, x):
            return self.resnet(x)
    