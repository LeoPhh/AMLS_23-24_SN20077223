import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class PneumoniaCNNClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_dataset()

    # Load the dataset and separate the training, validation and testing images/labels
    def load_dataset(self):
        pneumonia_dataset = np.load(self.dataset_path)
        self.training_images = self.normalize_images(pneumonia_dataset['train_images'])
        self.validation_images = self.normalize_images(pneumonia_dataset['val_images'])
        self.testing_images = self.normalize_images(pneumonia_dataset['test_images'])

        self.training_labels = pneumonia_dataset['train_labels']
        self.validation_labels = pneumonia_dataset['val_labels']
        self.testing_labels = pneumonia_dataset['test_labels']

        self.image_height, self.image_width = self.training_images[0].shape
        self.number_of_channels = 1  # Because they are grayscale images

    # Normalize images to have value between 0 and 1
    def normalize_images(self, images):
        return images.astype('float32') / 255.0

    # Build CNN model and add all the layers
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_height, self.image_width, self.number_of_channels)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Train the CNN model on the training data and validate it on the validation data
    def train_model(self, epochs, batch_size):
        self.model = self.build_model()
        self.model.fit(
            self.training_images, self.training_labels,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.validation_images, self.validation_labels)
        )

    # Evaluate the loss and the accuracy of the trained CNN model
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.testing_images, self.testing_labels)
        print(f'Test loss: {test_loss:.4f}')
        print(f'Test accuracy: {100*test_accuracy:.2f}%')


if __name__ == "__main__":
    classifier = PneumoniaCNNClassifier('./Datasets/pneumoniamnist.npz')
    classifier.train_model(10, 32)
    classifier.evaluate_model()
