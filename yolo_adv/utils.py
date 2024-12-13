import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class YOLOv8Dataloader(Dataset):
    def __init__(self, images_dir, annotations_dir, target_dir=None, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
       
        if target_dir:
            self.images = [
                file for file in os.listdir(images_dir)
                if not os.path.exists(os.path.join(target_dir, f"{os.path.splitext(file)[0]}_Pixle{os.path.splitext(file)[1]}"))
            ]
        else:
            self.images = [file for file in os.listdir(images_dir) if file.endswith('.jpg') or file.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
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
        
        sample = {'image': image, 'classes': classes, 'boxes': boxes, 'image_name': os.path.splitext(os.path.basename(img_name))[0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class YOLOv8DetectionLoss():
    def __init__(self, model, max_steps):
        self.model = model
        self.det_model = self.model.model
        self.det_model.args["box"] = 7.5
        self.det_model.args["cls"] = 0.5
        self.det_model.args["dfl"] = 1.5
        self.det_model.args["pose"] = 12.0
        self.det_model.args["kobj"] = 1.0
        
        self.det_model.init_criterion().proj = self.det_model.init_criterion().proj.to(device='cuda', non_blocking=True)
        self.losses = np.zeros(max_steps)
        
    def compute_loss(self, image, cls, box, step, get_logits=False, save_loss=True):
        
                
        # Resize the image to a consistent size
        resize_transform = transforms.Resize((736, 1280))  # Updated dimensions
        image = resize_transform(image)

        
        # image = torch.nn.functional.interpolate(image, size=(736, 1280), mode='bilinear', align_corners=False)
        
        batch = {'batch_idx': torch.randn_like(cls).to(device='cuda', non_blocking=True) ,'img': image.to(device='cuda', non_blocking=True),\
            'cls': cls.to(device='cuda', non_blocking=True) ,'bboxes': box.to(device='cuda', non_blocking=True)}
                                        
        tloss, _, logits = self.det_model.loss(preds=None, batch=batch)
        
        if save_loss:
            self.losses[step] = tloss.item()

        if get_logits:
            return tloss, logits
        else:
            return tloss