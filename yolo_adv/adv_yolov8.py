from torch.utils.data import DataLoader
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from utils import YOLOv8Dataloader
from torchattacks import PGD, FGSM, FFGSM, VNIFGSM, Pixle, DeepFool
from torchattacks.attacks.poltergeist import cal_blur
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import os


def plot_loss(dl_len, yolo_output, idx, col, atk):
    plt.suptitle(f"YOLOv8 Inference vs {atk.__repr__().split('(')[0]} Attack - Steps: {atk.steps if hasattr(atk,'steps') else 1}, Epsilon: {round(atk.eps,3)}, \
        Decay: {round(atk.decay,3) if hasattr(atk,'decay') else None}, Alpha: {round(atk.alpha,3) if hasattr(atk,'alpha') else None}, \
        Beta: {round(atk.beta,3) if hasattr(atk,'beta') else None}, N: {atk.N if hasattr(atk,'N') else None}")
    plt.subplot(dl_len, col, idx + 1).figure.set_size_inches(30, 15)
    plt.title("Inference Image")
    plt.axis('off')
    plt.imshow(yolo_output[0])
    plt.subplot(dl_len, col, idx + 2)
    plt.title("Adversarial Image")
    plt.axis('off')
    plt.imshow(yolo_output[1])
    if atk.__repr__().split('(')[0] == "PGD" or atk.__repr__().split('(')[0] == "VNIFGSM":
        plt.subplot(dl_len, col, idx + 3)
        plt.title("YOLOv8 Loss")
        plt.plot(loss)
        plt.grid()

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='YOLOv8 Attack')
    argparser.add_argument('--input_data_dir',
                           '-idd',
                           default='./yolo_adv/adv_data/norm',
                           type=str,
                           help='path to input data directory, must contain images and labels directories (default: ./yolo_adv/adv_data/norm)')
   
    argparser.add_argument('--output_data_dir',
                           '-odd',
                           default='./yolo_adv/adv_data/adv',
                           type=str,
                           help='path to output data directory (default: ./yolo_adv/adv_data/adv)')
    
    argparser.add_argument('--model_path',
                            '-mp',
                            default='./yolo_adv/best.pt',
                            type=str,
                            help='define the path of the YOLOv8 model to attack (default: ./yolo_adv/best.pt)')
        
    argparser.add_argument('--atk_type',
                           '-at',
                           default='PGD',
                           type=str,
                           help='select between PGD, FGSM, FFGSSM and VNIFGSM for your adversarial perturbation (default: PGD)')
    
    argparser.add_argument('--steps',
                            '-s',
                            default=100,
                            type=int,
                            help='define the number of steps of the adversarial attack, if applicable (default: 100)')
    
    argparser.add_argument('--plot',
                            '-p',
                            action='store_true',
                            default=False,
                            help='plot results (default: False)')
    
    argparser.add_argument('--save_inference',
                           '-si',
                           action='store_true',
                           default=False,
                           help='choose wheter to save inference results for both normal and adversarial images (default: False)')
    
    args = argparser.parse_args()
    
    transform = transforms.ToPILImage()

    model = YOLO(args.model_path)
    model.to(device='cuda', non_blocking=True)
    model.training = False

    if args.atk_type == 'PGD':
        norm_path = 'PGD'
        atk = PGD(model=model, yolo=True, eps=0.005, steps=args.steps)
    elif args.atk_type == 'FGSM':
        atk = FGSM(model=model, yolo=True, eps=0.024)
    elif args.atk_type == 'FFGSM':
        norm_path = 'F-FGSM'
        atk = FFGSM(model=model, yolo=True, eps=0.0024, alpha=0.0055)
    elif args.atk_type == 'VNIFGSM':
        norm_path = 'VNI-FGSM'
        atk = VNIFGSM(model=model, yolo=True, eps=0.0024, alpha=0.005, steps=args.steps, decay=1.0, N=5, beta=3/2)
    elif args.atk_type == 'PIXLE':
        norm_path = 'Pixle'
        atk = Pixle(model, yolo=True, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=60, update_each_iteration=True)
    elif args.atk_type == 'DEEPFOOL':
        norm_path = 'DeepFool'
        atk = DeepFool(model, yolo=True, steps=10, overshoot=0.002)
    elif args.atk_type == 'POLTER':
        norm_path = 'Poltergeist'
    
    
    
    # Create the dataset
    dataset = YOLOv8Dataloader(images_dir=f'{args.input_data_dir}/images', annotations_dir=f'{args.input_data_dir}/labels', transform=None)
 
    # Create the DataLoader
    data_loader = DataLoader(dataset, shuffle=False, collate_fn=lambda x: x)
    dl_len = len(data_loader)
     
    for i, data in tqdm(enumerate(data_loader), total=dl_len, desc=f"Running {atk.__repr__().split('(')[0]} Attack"):

        yolo_output = []
        if norm_path != 'Poltergeist':
            loss, adv_img = atk(data[0]['image'], data[0]['classes'], data[0]['boxes'])
        else:
            adv_img = cal_blur(data[0]['image'][0].detach().clone(), 1.5, 0, 0)
        inference_img = transform(data[0]['image'][0].detach().clone())
        adv_img = transform(adv_img[0].detach().clone())
                
        results = model([inference_img, adv_img], verbose=False)
        
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            yolo_output.append(im)

        if args.plot:
            if atk.__repr__().split('(')[0] == "PGD" or atk.__repr__().split('(')[0] == "VNIFGSM":
                plot_loss(dl_len, yolo_output, i*3, 3, atk)
            else:
                plot_loss(dl_len, yolo_output, i*2, 2, atk)        
        
        directories = ['adv_img', 'norm_inf', 'adv_inf']
        for directory in directories:
            os.makedirs(f'{args.output_data_dir}/{directory}/', exist_ok=True)
            os.makedirs(f'{args.output_data_dir}/{directory}/{atk.__repr__().split("(")[0]}/', exist_ok=True)
            os.makedirs(f'{args.output_data_dir}/{directory}/{atk.__repr__().split("(")[0]}/images/', exist_ok=True)
            if not args.save_inference:
                break
        
        adv_img.save(f'{args.output_data_dir}/adv_img/{atk.__repr__().split("(")[0]}/images/{data[0]["image_name"]}_{atk.__repr__().split("(")[0]}.png')
        if args.save_inference:
            for idx, output in enumerate(yolo_output):
                output.save(f'{args.output_data_dir}/{directories[idx+1]}/{atk.__repr__().split("(")[0]}/images/{data[0]["image_name"]}_{atk.__repr__().split("(")[0]}.png')
            
    plt.show()