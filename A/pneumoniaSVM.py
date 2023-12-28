import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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

    # Method to extract Pixel Values and Histograms as features from an array of images
    def extract_features(self, images):
        # Flatten pixel values for each image
        pixel_values = images.reshape(images.shape[0], -1)

        # Extract Histograms
        histograms = np.array([np.histogram(image.flatten(), bins=256, range=[0, 256])[0] for image in images])

        # Return combined features
        return np.concatenate((pixel_values, histograms), axis=1)

    # Method to display a sample histogram for the report
    def display_sample_histogram(self):
        histograms = np.array([np.histogram(image.flatten(), bins=256, range=[0, 256])[0] for image in self.training_images])
        first_image_histogram = histograms[0]
        plt.figure(figsize=(8, 4))
        plt.bar(range(256), first_image_histogram, width=1, color='gray')
        plt.title('Histogram for the First Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    # Method to train the SVM model
    def train_model(self):
        # Extract features from training images
        training_data_features = self.extract_features(self.training_images)

        # Train the model
        self.svm_model = SVC(kernel='linear')
        self.svm_model.fit(training_data_features, self.training_labels.ravel())

    # Method to validate and test the SVM model
    def validate_and_test_model(self):
        # Extract features from validation and testing images
        validation_data_features = self.extract_features(self.validation_images)
        testing_data_features = self.extract_features(self.testing_images)

        # Get validation and testing predictions
        validation_predictions = self.svm_model.predict(validation_data_features)
        testing_predictons = self.svm_model.predict(testing_data_features)

        # Evaluate on validation set
        val_accuracy = accuracy_score(self.validation_labels, validation_predictions)
        print(f'Validation Accuracy: {100*val_accuracy:.2f}%')

        # Evaluate on test set
        test_accuracy = accuracy_score(self.testing_labels, testing_predictons)
        print(f'Test Accuracy: {100*test_accuracy:.2f}%')

        # Display classification report for test set
        print("\nClassification Report:")
        print(classification_report(self.testing_labels, testing_predictons))


if __name__ == "__main__":
    classifier = PneumoniaSVMClassifier('./Datasets/pneumoniamnist.npz')
    classifier.train_model()
    classifier.validate_and_test_model()
