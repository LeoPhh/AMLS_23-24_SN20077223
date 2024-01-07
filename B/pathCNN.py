from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


class PathCNNClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_dataset()

    # Load the dataset and concatenate training and validation data
    def load_dataset(self):
        path_dataset = np.load(self.dataset_path)

        # Training images and labels
        self.training_images = self.normalize_images(path_dataset['train_images'])
        self.training_labels = to_categorical(path_dataset['train_labels'], num_classes=9)

        # Validation images and labels
        self.validation_images = self.normalize_images(path_dataset['val_images'])
        self.validation_labels = to_categorical(path_dataset['val_labels'], num_classes=9)

        # Testing images and labels
        self.testing_images = self.normalize_images(path_dataset['test_images'])
        self.testing_labels = to_categorical(path_dataset['test_labels'], num_classes=9)

        self.image_height, self.image_width, self.number_of_channels = self.training_images[0].shape

    # Method to print the unique labels and their count
    def print_individual_label_counts(self):
        # Need to load training labels again so that they are not one-hot-encoded
        path_dataset = np.load(self.dataset_path)
        training_labels = path_dataset['train_labels']

        # Use numpy to return two arrays: one for the unique values, and another for their count
        unique_values, counts = np.unique(training_labels.flatten(), return_counts=True)

        print("Individual label counts:")

        # Display the unique values and their count by zipping the two arrays
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count}")

    # Method to normalize pixel values in images between 0 and 1
    def normalize_images(self, images):
        return images.astype('float32') / 255.0

    # Build CNN model and add all the layers
    def build_model(self):
        model = Sequential()
    
        # Hidden layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_height, self.image_width, self.number_of_channels)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))  # Introduce dropout for regularization
        model.add(Dense(128, activation='relu'))
        model.add(Dense(9, activation='softmax'))

        # Compile and return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Train the CNN model without cross-validation
    def train_without_cross_validation(self, epochs, batch_size):
        print("Training Task B CNN without cross-validation...")

        # Early stopping callback to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Build model and train it with training and validation datasets
        self.model = self.build_model()
        self.history = self.model.fit(
            self.training_images,
            self.training_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.validation_images, self.validation_labels)
            # callbacks=[early_stopping]
        )

        # Evaluate the model through the test accuracy
        self.test_model(used_cross_validation=False)

    # Method to train the CNN model with cross-validation
    def train_with_cross_validation(self, epochs, batch_size, folds):
        print("Training Task B CNN with cross-validation...")

        # Concatenate training and validation data for cross validation
        training_images = np.concatenate((self.training_images, self.validation_images), axis=0)
        training_labels = np.concatenate((self.training_labels, self.validation_labels), axis=0)

        # Empty array to store all the models for each fold
        self.models = []

        # Prepare cross validation folds with shuffling
        cross_validation = KFold(n_splits=folds, shuffle=True)
        cross_validation_folds = cross_validation.split(self.training_images, self.training_labels)

        # Iterate through the folds and train the model
        for fold, (train_index, val_index) in enumerate(cross_validation_folds):
            print(f"Training on fold {fold + 1}/{folds}")

            # Create training images dataset for the current fold
            current_fold_training_images = training_images[train_index]
            current_fold_training_labels = training_labels[train_index]

            # Create validation images dataset for current fold
            current_fold_validation_images = training_images[val_index]
            current_fold_validation_labels = training_labels[val_index]

            # Early stopping callback to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            # Train model on each individual fold (using the same validation data), and append it to the models array
            self.model = self.build_model()
            self.model.fit(
                current_fold_training_images,
                current_fold_training_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(current_fold_validation_images, current_fold_validation_labels),
                callbacks=[early_stopping]
            )

            self.models.append(self.model)

        # Evaluate the model through the test accuracy
        self.test_model(used_cross_validation=True)
    
    # Method to evaluate the loss and the accuracy of the trained CNN model
    def test_model(self, used_cross_validation):
        # Calculate average accuracy of all models if cross validation was used
        if used_cross_validation:
            # Empty array to store the ensemble predictions (predictions from all models)
            ensemble_predictions = []

            # Iterate through all the models and obtain their predictions
            for model in self.models:
                current_fold_predictions = model.predict(self.testing_images, verbose=0)
                ensemble_predictions.append(current_fold_predictions)

            # Calculate the column-wise average of each prediction i.e. the average prediction for each image, from all the models
            averaged_predictions = np.mean(ensemble_predictions, axis=0)

            # Calculate the accuracy
            accuracy = self.get_accuracy(averaged_predictions)

            print(f'Averaged test accuracy of all models: {100 * accuracy:.2f}%')


        # Else calculate accuracy of just the one model if cross validation was not used
        else:
            # Calculate accuracy using same methodology from above
            predictions = self.model.predict(self.testing_images, verbose=0)
            accuracy = self.get_accuracy(predictions)

            print(f'Test accuracy of one model: {100 * accuracy:.2f}%')

    # Method to get model accuracy by comparing predictions with true testing labels
    def get_accuracy(self, predictions):
        # Round the one-hot encoded predictions
        rounded_predictions = np.round(predictions)
        correct = 0
        total = len(self.testing_labels)

        # Iterate through the rounded predictions and compare with corresponding true label
        for i in range(total):
            # Convert one-hot encoded values into a single number corresponding to the class
            # for example [0 0 0 1] becomes 3
            converted_prediction = np.argmax(rounded_predictions[i], axis=None, out=None)
            converted_label = np.argmax(self.testing_labels[i], axis=None, out=None)
            # Add one to the correct count if there is a match
            if converted_prediction == converted_label:
                correct += 1
        
        # Return accuracy
        return (correct/total)
    
    # Method to plot the training and validation accuracy of the model
    # This method can ONLY be used when cross validation is NOT used! Otherwise there will be no self.history variable
    def plot_training_process(self):
        model_history = self.history

        # Plot training & validation accuracy
        plt.figure(figsize=(6, 4))
        plt.plot(model_history.history['accuracy'])
        plt.plot(model_history.history['val_accuracy'])
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.grid()
        plt.show()

        # Plot training & validation loss
        plt.figure(figsize=(6, 4))
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.grid()
        plt.show()