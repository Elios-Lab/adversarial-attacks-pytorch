import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class YOLOv8Dataloader(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # List of filenames of images
        self.images = [file for file in os.listdir(images_dir) if file.endswith('.jpg') or file.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
                
        # Resize the image to a consistent size
        resize_transform = transforms.Resize((480, 640))
        image = resize_transform(image)
        
        image = transforms.ToTensor()(image)
        image = image[None, :, :, :]

        # Load annotations
        annot_name = os.path.join(self.annotations_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes, classes = [], []
        with open(annot_name, 'r') as file:
            for line in file:
                class_label, x_center, y_center, width, height = map(float, line.split())
                class_tensor = torch.tensor(class_label) 
                b_tensor = torch.tensor([x_center, y_center, width, height])
                boxes.append(b_tensor)
                classes.append(class_tensor)
        
        if boxes != [] or classes != []:
            boxes = torch.stack(boxes)
            classes = torch.stack(classes)
        else:
            boxes = torch.zeros((0, 4))
            classes = torch.zeros((0, 1))
        
        sample = {'image': image, 'classes': classes, 'boxes': boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample


class YOLOv8DetectionLoss():
    def __init__(self, model, max_steps):
        self.model = model
        self.det_model = self.model.model
        self.det_model.criterion.proj = self.det_model.criterion.proj.to(device='cuda', non_blocking=True)
        self.losses = np.zeros(max_steps)
        
    def compute_loss(self, image, cls, box, step, get_logits=False, requires_grad=False, save_loss=True):
        batch = {'batch_idx': torch.randn_like(cls).to(device='cuda', non_blocking=True) ,'img': image.to(device='cuda', non_blocking=True),\
            'cls': cls.to(device='cuda', non_blocking=True) ,'bboxes': box.to(device='cuda', non_blocking=True)}  
                        
        tloss, _, logits = self.det_model.loss(preds=None, batch=batch)
        if save_loss:
            self.losses[step] = tloss.item()
        tloss.requires_grad_(requires_grad)
        if get_logits:
            return tloss, logits
        else:
            return tloss