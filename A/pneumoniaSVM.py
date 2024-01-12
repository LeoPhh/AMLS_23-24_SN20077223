import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report


class PneumoniaSVMClassifier:
    def __init__(self, dataset_path):
        self.load_dataset(dataset_path)

    # Method to load the dataset and separate the training, validation and testing images/labels
    def load_dataset(self, dataset_path):
        pneumonia_dataset = np.load(dataset_path)

        # Create variables for the images
        self.training_images = pneumonia_dataset['train_images']
        self.validation_images = pneumonia_dataset['val_images']
        self.testing_images = pneumonia_dataset['test_images']

        # Create variables for the labels
        self.training_labels = pneumonia_dataset['train_labels']
        self.validation_labels = pneumonia_dataset['val_labels']
        self.testing_labels = pneumonia_dataset['test_labels']

    # Method to print the unique labels and their count
    def print_individual_label_counts(self):
        # Use numpy to return two arrays: one for the unique values, and another for their count
        unique_values, counts = np.unique(self.training_labels.flatten(), return_counts=True)

        print("Individual label counts:")

        # Display the unique values and their count by zipping the two arrays
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count}")

    # Method to extract Pixel Values and Histograms as features from an array of images
    def extract_features(self, images):
        # Extract Pixel Values
        pixel_values = images.reshape(images.shape[0], -1)

        # Extract Histograms
        histograms = np.array([np.histogram(image.flatten(), bins=256, range=[0, 256])[0] for image in images])

        # Return combined features
        return np.concatenate((pixel_values, histograms), axis=1)

    # Method to display a sample image and histogram for the report
    def display_sample_histogram(self, sample_image_index=0):
        # Extract one image and its corresponding histogram
        sample_image = self.training_images[sample_image_index]
        sample_image_histogram = np.histogram(sample_image.flatten(), bins=256, range=[0, 256])[0]

        # Create plot figure
        plt.figure(figsize=(12, 4))

        # Subplot for the image
        plt.subplot(1, 2, 1)
        plt.imshow(sample_image, cmap='gray')
        plt.title('Sample (Grayscale) Image')

        # Subplot for the histogram
        plt.subplot(1, 2, 2)
        plt.bar(range(256), sample_image_histogram, width=1)
        plt.title('Corresponding Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.show()

    # Method to train the SVM model
    def train_model(self):
        print("Training Task A SVM model...")

        # Extract features from training and validation images
        training_data_features = self.extract_features(self.training_images)
        validation_data_features = self.extract_features(self.validation_images)

        # Empty array to store the hyperparameter combinations and validation accuracy
        parameters_and_accuracy = []

        # Hyperparameters to be tuned
        regularization_parameters = [0.1, 1, 10]
        kernels = ['linear', 'poly', 'sigmoid']

        # Iterate through all possible hyperparameter combinations and train the SVM model
        for regularization_parameter in regularization_parameters:
            for kernel in kernels:
                temporary_svm_model = svm.SVC(C=regularization_parameter, kernel=kernel)
                temporary_svm_model.fit(training_data_features, self.training_labels.ravel())

                # Evaluate the model on the validation dataset
                validation_predictions = temporary_svm_model.predict(validation_data_features)
                validation_accuracy = accuracy_score(self.validation_labels, validation_predictions)

                # Store the hyperparameter combination along with the validation accuracy
                current_model_results = [regularization_parameter, kernel, validation_accuracy]
                parameters_and_accuracy.append(current_model_results)

        # Extract the best model by finding the one with the highest validation accuracy
        best_model = max(parameters_and_accuracy, key=lambda x: x[2])
        best_model_regularization_parameter = best_model[0]
        best_model_kernel = best_model[1]

        # Train the model with the optimal parameters
        self.svm_model = svm.SVC(C=best_model_regularization_parameter, kernel=best_model_kernel)
        self.svm_model.fit(training_data_features, self.training_labels.ravel())

        print("Finished training.")


    # Method to validate and test the SVM model
    def test_model(self):
        # Extract features from testing images and get testing predictions
        testing_data_features = self.extract_features(self.testing_images)
        testing_predictons = self.svm_model.predict(testing_data_features)

        # Evaluate on test set
        test_accuracy = accuracy_score(self.testing_labels, testing_predictons)
        print(f'Test Accuracy: {100*test_accuracy:.2f}%')

        # Display classification report for test set
        print("\nClassification Report:")
        print(classification_report(self.testing_labels, testing_predictons))

    