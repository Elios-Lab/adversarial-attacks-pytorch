import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

class Classifier():
    def __init__(self):
        self.model = self.BinaryResNet50()
        self.current_epoch = 0
        self.exp_name = None
        self.optimizer = None
        self.class_names = {0: 'adv', 1: 'real'}
        # Check if GPU (CUDA) is available
        if tf.config.list_physical_devices('GPU'):
            print("GPU is available. Using GPU.")
        else:
            print("GPU is not available. Using CPU.")

    def load_dataset(self, root, normalize=True, batch_size=16, shuffle=True):
        if not os.path.exists(root):
            raise FileNotFoundError(f"The specified path does not exist: {root}")

        # ImageDataGenerator for data loading and augmentation
        if normalize:
            datagen = ImageDataGenerator(
                rescale=1.0/255,
                validation_split=0.2,
                preprocessing_function=tf.keras.applications.resnet50.preprocess_input
            )
        else:
            datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

        # Load training and validation datasets
        self.train_loader = datagen.flow_from_directory(
            root,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=shuffle
        )

        self.valid_loader = datagen.flow_from_directory(
            root,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=shuffle
        )

    def train(self, exp_name='exp', epochs=10, lr=0.001, valid_period=5, ckpt_period=None, patience=None):
        self.exp_name = exp_name
        checkpoint_path = f'./runs/{self.exp_name}/ckpts'
        os.makedirs(checkpoint_path, exist_ok=True)

        # Compile the model
        self.model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Callbacks for saving checkpoints and early stopping
        callbacks_list = [
            callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, "epoch-{epoch:02d}.h5"),
                                      save_weights_only=True,
                                      period=ckpt_period if ckpt_period else epochs + 1),
            callbacks.TensorBoard(log_dir=f'./runs/{self.exp_name}/tb_logs')
        ]
        if patience:
            callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))

        # Train the model
        self.model.fit(
            self.train_loader,
            epochs=epochs,
            validation_data=self.valid_loader,
            validation_freq=valid_period,
            callbacks=callbacks_list
        )

        # Save the final model
        self.model.save_weights(f'./runs/{self.exp_name}/last.h5')

    def validate(self):
        self.model.evaluate(self.valid_loader)

    def evaluate(self, valid_loader=None):
        if valid_loader is None:
            valid_loader = self.valid_loader
        results = self.model.evaluate(valid_loader)
        print(f'Validation loss: {results[0]}, Validation accuracy: {results[1]}')

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def predict(self, image_path=None, image: Image = None):
        image = self.process_image(image_path=image_path, image=image)
        image = np.expand_dims(image, axis=0)

        # Perform inference
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return self.class_names[predicted_class]

    def process_image(self, image_path=None, image: Image = None):
        if image is None:
            image = Image.open(image_path)
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image

    class BinaryResNet50(tf.keras.Model):
        def __init__(self):
            super(Classifier.BinaryResNet50, self).__init__()
            base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(2, activation='softmax')
            ])

        def call(self, inputs):
            return self.model(inputs)