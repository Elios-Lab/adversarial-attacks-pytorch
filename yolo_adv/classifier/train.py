from classifier import Classifier

if __name__ == "__main__":
    data_path = '/home/elios/pighetti/adversarial-attacks-pytorch/yolo_adv/classifier/data'
    model = Classifier()
    model.load_dataset(root=data_path, normalize=True, batch_size=16, shuffle=True)
    model.train(epochs=33, lr=0.001, valid_period=1, ckpt_period=5)
    model.evaluate()
    