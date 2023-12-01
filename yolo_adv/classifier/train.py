from classifier import Classifier

if __name__ == "__main__":
    data_path = 'C:/Users/luca/Desktop/adv'
    model = Classifier()
    model.load_dataset(root=data_path, normalize=True, batch_size=16, shuffle=True)
    model.train(epochs=2, lr=0.001, valid_period=5, ckpt_period=None)
    model.evaluate()
    