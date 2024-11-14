from classifier import Classifier

if __name__ == "__main__":
    data_path = '/home/elios/lazzaroni/adv/datasets/FC_5K_3K_3K'  # r'C:\Users\lazzaroni\Documents\adv\datasets\FC'
    model = Classifier('mobilenetv2')
    model.load_dataset(root=data_path, batch_size=256, shuffle=True)
    model.train(exp_name='3_classes_mn', epochs=100, lr=0.001, valid_period=1, ckpt_period=5, patience=5, amp=True)
    model.evaluate(amp=True)