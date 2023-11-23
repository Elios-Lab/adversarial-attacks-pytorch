from torch.utils.data import DataLoader
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from utils import YOLOv8Dataloader, YOLOv8DetectionLoss
from torchattacks import PGD, FGSM, FFGSM, VNIFGSM
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2


def plot_loss(idx, col, atk):
    plt.suptitle(f"YOLOv8 Inference vs {atk.__repr__().split('(')[0]} Attack - Steps: {atk.steps if hasattr(atk,'steps') else 1}, Epsilon: {round(atk.eps,3)}, Decay: {round(atk.decay,3) if hasattr(atk,'decay') else None}, Alpha: {round(atk.alpha,3) if hasattr(atk,'alpha') else None}, Beta: {round(atk.beta,3) if hasattr(atk,'beta') else None}, N: {atk.N if hasattr(atk,'N') else None}")
    plt.subplot(len(data_loader), col, idx + 1).figure.set_size_inches(20, 10)
    plt.title("Inference Image")
    plt.axis('off')
    plt.imshow(yolo_output[0])
    plt.subplot(len(data_loader), col, idx + 2)
    plt.title("Adversarial Image")
    plt.axis('off')
    plt.imshow(yolo_output[1])
    if atk.__repr__().split('(')[0] == "PGD" or atk.__repr__().split('(')[0] == "VNIFGSM":
        plt.subplot(len(data_loader), col, idx + 3)
        plt.title("YOLOv8 Loss")
        plt.plot(loss)
        plt.grid()

if __name__ == '__main__':
    
    transform = transforms.ToPILImage()

    model = YOLO("./yolo_adv/best.pt")
    model.to(device='cuda', non_blocking=True)
    model.training = False

    steps = 100
     
    atk = PGD(model=model, yolo=True, eps=0.024, steps=steps)
    # atk = FGSM(model=model, yolo=True, eps=0.024)
    # atk = FFGSM(model=model, yolo=True, eps=0.024, alpha=0.055)
    # atk = VNIFGSM(model=model, yolo=True, eps=0.024, alpha=0.05, steps=steps, decay=1.0, N=5, beta=3/2)
    # atk = DeepFool(model=model, yolo=True, steps=steps)
    # 
    # Create the dataset
    dataset = YOLOv8Dataloader(images_dir='./yolo_adv/test_data/images', annotations_dir='./yolo_adv/test_data/labels/', transform=None)
 
    # Create the DataLoader
    data_loader = DataLoader(dataset, shuffle=True, collate_fn=lambda x: x)
   
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Running {atk.__repr__().split('(')[0]} Attack"):
        yolo_output = []
        loss, adv_img = atk(data[0]['image'], data[0]['boxes'], data[0]['classes'])
    
        inference_img = transform(data[0]['image'][0].detach().clone())
        adv_img = transform(adv_img[0].detach().clone())
        
        results = model([inference_img, adv_img], verbose=False)
        
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            yolo_output.append(im)
        # 
        if atk.__repr__().split('(')[0] == "PGD" or atk.__repr__().split('(')[0] == "VNIFGSM":
            plot_loss(i*3, 3, atk)
        else:
            plot_loss(i*2, 2, atk)
            
    plt.show()
