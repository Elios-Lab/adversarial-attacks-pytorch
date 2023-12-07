from classifier import Classifier

if __name__ == "__main__":
    img_path = '/home/pigo/projects/adversarial-attacks-pytorch/yolo_adv/adv_data/norm/images/dayClip7--00930.jpg'
    # ds_path = 'C:/Users/luca/Desktop/adv_bosch/test/VALEO/PGD'
    classifier = Classifier()
    classifier.load_model(model_path='yolo_adv/classifier/last.pt')
    print(classifier.predict(image_path=img_path))
    # print(classifier.evaluate_model_on_dataset(dataset_path=ds_path))
    