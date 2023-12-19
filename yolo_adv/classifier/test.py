from classifier import Classifier

if __name__ == "__main__":
    # img_path = '/home/pigo/projects/adversarial-attacks-pytorch/yolo_adv/adv_data/norm/images/dayClip7--00930.jpg'
    ds_path = '/Users/luca/Documents/adv_cls_data/test_data'
    classifier = Classifier()
    classifier.load_model(model_path='yolo_adv/classifier/runs/exp/last.pt')
    #./yolo_adv/classifier/runs/exp/luca_last.pt
    # print(classifier.predict(image_path=img_path))
    result = classifier.evaluate_model_on_dataset(dataset_path=ds_path)
    print(f'Accuracy: {result[0]}')
    print(f'Precision: {result[1]}')    
    print(f'Recall: {result[2]}')
    print(f'F1 Score: {result[3]}')
    