from classifier import Classifier

if __name__ == "__main__":
    # img_path = '/home/pigo/projects/adversarial-attacks-pytorch/yolo_adv/adv_data/norm/images/dayClip7--00930.jpg'
    ds_path = '/home/elios/lazzaroni/adv/FC_2K_500_1K/test'  # r'C:\Users\lazzaroni\Documents\adv\datasets\FC\test'
    classifier = Classifier('mobilenetv2')
    classifier.load_model(model_path='yolo_adv/classifier/runs/3_classes/last.pt')
    #./yolo_adv/classifier/runs/exp/luca_last.pt
    # print(classifier.predict(image_path=img_path))
    result = classifier.evaluate_model_on_dataset(dataset_path=ds_path, amp=True)
    print(f'Accuracy: {result[0]}')
    print(f'Precision: {result[1]}')    
    print(f'Recall: {result[2]}')
    print(f'F1 Score: {result[3]}')
    print(f'True Positive Rate: {result[4]}')
    print(f'False Positive Rate: {result[5]}')
    print(f'True Negative Rate: {result[6]}')
    print(f'False Negative Rate: {result[7]}')
    classifier.plot_cm(result[8])