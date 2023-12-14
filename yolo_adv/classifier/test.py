from classifier import Classifier

if __name__ == "__main__":
    # img_path = '/home/pigo/projects/adversarial-attacks-pytorch/yolo_adv/adv_data/norm/images/dayClip7--00930.jpg'
    ds_path = './yolo_adv/cls_test_data'
    classifier = Classifier()
    classifier.load_model(model_path='/home/elios/pighetti/adversarial-attacks-pytorch/yolo_adv/classifier/runs/exp/last.pt')
    #./yolo_adv/classifier/runs/exp/luca_last.pt
    # print(classifier.predict(image_path=img_path))
    print(classifier.evaluate_model_on_dataset(dataset_path=ds_path))
    