import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import StratifiedKFold

class PneumoniaCNNClassifier:
    def __init__(self, dataset_path):
        self.load_dataset(dataset_path)

    # Method to load the dataset and separate the training, validation and testing images/labels
    def load_dataset(self, dataset_path):
        pneumonia_dataset = np.load(dataset_path)

        # Create variables for the normalised images
        self.training_images = self.normalize_images(pneumonia_dataset['train_images'])
        self.validation_images = self.normalize_images(pneumonia_dataset['val_images'])
        self.testing_images = self.normalize_images(pneumonia_dataset['test_images'])

        # Create variables for the labels
        self.training_labels = pneumonia_dataset['train_labels']
        self.validation_labels = pneumonia_dataset['val_labels']
        self.testing_labels = pneumonia_dataset['test_labels']

        # Extract the image height, width and number of channels to be used in the CNN
        self.image_height, self.image_width = self.training_images[0].shape
        self.number_of_channels = 1  # Because they are grayscale images

    # Method to normalize images to have value between 0 and 1
    def normalize_images(self, images):
        return images.astype('float32') / 255.0

    # Method to build CNN model and add all the layers
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_height, self.image_width, self.number_of_channels)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Method to train the CNN model without cross-validation
    def train_without_cross_validation(self, epochs, batch_size):
        print("Training without cross-validation")

        # Build model and train it with training and validation datasets
        self.model = self.build_model()
        self.model.fit(
            self.training_images,
            self.training_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.validation_images, self.validation_labels)
        )

        # Evaluate the model through the test accuracy
        self.test_model(used_cross_validation=False)

    # Method to train the CNN model with cross-validation
    def train_with_cross_validation(self, epochs, batch_size, folds):
        print("Training with cross-validation")

        # To store all the models for each fold
        self.models = []

        # Prepare cross validation folds with shuffling
        cross_validation = StratifiedKFold(n_splits=folds, shuffle=False)
        cross_validation_folds = cross_validation.split(self.training_images, self.training_labels)

        # In the tuple (train_index, val_index), we don't use the val_index because every fold uses the same validation dataset
        for fold, (train_index, val_index) in enumerate(cross_validation_folds):
            print(f"Training on fold {fold + 1}/{folds}")
            current_fold_images = self.training_images[train_index]
            current_fold_labels = self.training_labels[train_index]

            # Train model on each individual fold (using the same validation data), and append it to the models array
            self.model = self.build_model()
            self.model.fit(
                current_fold_images,
                current_fold_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.validation_images, self.validation_labels)
            )
            self.models.append(self.model)

        # Evaluate the model through the test accuracy
        self.test_model(used_cross_validation=True)


    # Evaluate the loss and the accuracy of the trained CNN model
    def test_model(self, used_cross_validation):
        # Print accuracies of all models if cross validation was used
        if used_cross_validation:
            for fold, model in enumerate(self.models):
                test_loss, test_accuracy = model.evaluate(self.testing_images, self.testing_labels, verbose=0)
                print(f'Fold {fold + 1} - Test Loss: {test_loss:.4f}, Test accuracy: {100 * test_accuracy:.2f}%')
        # Else print accuracy of just the one model if cross validation was not used
        else:
            test_loss, test_accuracy = self.model.evaluate(self.testing_images, self.testing_labels, verbose=0)
            print(f'Test loss: {test_loss:.4f}, Test accuracy: {100 * test_accuracy:.2f}%')


if __name__ == "__main__":
    # Example usage with cross-validation
    classifier_with_cross_val = PneumoniaCNNClassifier('./Datasets/pneumoniamnist.npz')
    classifier_with_cross_val.train_with_cross_validation(epochs=20, batch_size=32, folds=10)
