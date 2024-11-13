from classifier import Classifier

if __name__ == "__main__":
    data_path = '/home/elios/lazzaroni/adv/FC_2K_500_1K'  # r'C:\Users\lazzaroni\Documents\adv\datasets\FC'
    model = Classifier('mobilenetv2')
    model.load_dataset(root=data_path, batch_size=128, shuffle=True)
    model.train(exp_name='3_classes', epochs=50, lr=0.0005, valid_period=1, ckpt_period=5, patience=5, amp=True)
    model.evaluate(amp=True)