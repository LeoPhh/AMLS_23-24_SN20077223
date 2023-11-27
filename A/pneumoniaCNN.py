import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class PneumoniaCNNClassifier:
    def __init__(self, dataset_path, use_cross_validation, use_bagging, folds = None):
        self.dataset_path = dataset_path
        self.folds = folds
        self.use_cross_validation = use_cross_validation
        self.use_bagging = use_bagging
        self.models = []
        self.load_dataset()

    # Load the dataset and concatenate training and validation data
    def load_dataset(self):
        pneumonia_dataset = np.load(self.dataset_path)

        # Training images and labels
        self.training_images = self.normalize_images(pneumonia_dataset['train_images'])
        self.training_labels = pneumonia_dataset['train_labels']

        # Validation images and labels
        self.validation_images = self.normalize_images(pneumonia_dataset['val_images'])
        self.validation_labels = pneumonia_dataset['val_labels']

        # Testing images and labels
        self.testing_images = self.normalize_images(pneumonia_dataset['test_images'])
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

    # Train the CNN model on the combined training and validation data and validate it on the cross-validated folds
    def train_and_test_model(self, epochs, batch_size):
        if self.use_cross_validation:
            self.train_with_cross_validation(epochs, batch_size)
        else:
            self.train_without_cross_validation(epochs, batch_size)

    # Train the CNN model without cross-validation
    def train_without_cross_validation(self, epochs, batch_size):
        print("Training without cross-validation")
        self.model = self.build_model()
        self.model.fit(
            self.training_images, self.training_labels,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.validation_images, self.validation_labels)
        )

    # Train the CNN model with cross-validation
    def train_with_cross_validation(self, epochs, batch_size):
        print("Training with cross-validation")
        cross_validation = StratifiedKFold(n_splits=self.folds, shuffle=True)
        cross_validation_folds = cross_validation.split(self.training_images, self.training_labels)

        # In the tuple (train_index, _), we don't use the val_index because every fold validates on the same validation dataset provided
        for fold, (train_index, _) in enumerate(cross_validation_folds):
            print(f"Training on fold {fold + 1}/{self.folds}")
            current_fold_images = self.training_images[train_index]
            current_fold_labels = self.training_labels[train_index]
            self.model = self.build_model()
            self.model.fit(
                current_fold_images, current_fold_labels,
                epochs=epochs, batch_size=batch_size,
                validation_data=(self.validation_images, self.validation_labels)
            )

            self.models.append(self.model)

        if self.use_bagging:
            self.evaluate_model_with_bagging()
        else:
            self.evaluate_model()

    # Evaluate the loss and the accuracy of the trained CNN model
    def evaluate_model(self):
        if self.use_cross_validation:
            for fold, model in enumerate(self.models):
                test_loss, test_accuracy = model.evaluate(self.testing_images, self.testing_labels, verbose=0)
                print(f'Fold {fold + 1} - Test accuracy: {100 * test_accuracy:.2f}%')
        else:
            test_loss, test_accuracy = self.model.evaluate(self.testing_images, self.testing_labels)
            print(f'Test accuracy: {100 * test_accuracy:.2f}%')

    # Method to perform bagging and aggregate predictions
    def bagging(self, test_data):
        # Aggregate predictions from each model
        predictions = np.zeros((len(test_data),))
        for model in self.models:
            model_predictions = model.predict(test_data)
            predictions += model_predictions.flatten()

        # Take the majority vote as the final prediction
        predictions = (predictions >= (len(self.models) / 2)).astype(int)

        return predictions

    # Evaluate the total accuracy of the trained CNN model using bagging
    def evaluate_model_with_bagging(self):
        # Display test accuracy of each CV fold
        for fold, model in enumerate(self.models):
            test_predictions = model.predict(self.testing_images)
            test_accuracy = accuracy_score(self.testing_labels, (test_predictions >= 0.5).astype(int))
            print(f'Fold {fold + 1} - Test accuracy: {100 * test_accuracy:.2f}%')

        # Display test accuracy with bagging
        bagged_predictions = self.bagging(self.testing_images)
        total_accuracy = accuracy_score(self.testing_labels, bagged_predictions)
        print(f'Total accuracy with bagging: {100 * total_accuracy:.2f}%')



if __name__ == "__main__":
    # Example usage with cross-validation
    classifier_with_cross_val = PneumoniaCNNClassifier(
        '../Datasets/pneumoniamnist.npz', 
        use_cross_validation = True, 
        use_bagging = False,
        folds = 10
        )
    classifier_with_cross_val.train_and_test_model(20, 32)
