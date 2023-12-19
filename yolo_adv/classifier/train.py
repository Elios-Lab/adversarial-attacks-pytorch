from classifier import Classifier

if __name__ == "__main__":
    data_path = '/Users/luca/Documents/adv_cls_data/train_data'
    model = Classifier()
    model.load_dataset(root=data_path, normalize=True, batch_size=128, shuffle=True)
    model.train(exp_name='lr_exp', epochs=20, lr=0.0005, valid_period=1, ckpt_period=3, patience=5)
    model.evaluate()
    