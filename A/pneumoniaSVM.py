import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage import feature
import matplotlib.pyplot as plt

class PneumoniaSVMClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_dataset()

    # Load the dataset and separate the training, validation and testing images/labels
    def load_dataset(self):
        pneumonia_dataset = np.load(self.dataset_path)
        self.training_images = pneumonia_dataset['train_images']
        self.validation_images = pneumonia_dataset['val_images']
        self.testing_images = pneumonia_dataset['test_images']

        self.training_labels = pneumonia_dataset['train_labels']
        self.validation_labels = pneumonia_dataset['val_labels']
        self.testing_labels = pneumonia_dataset['test_labels']

    def extract_features(self, images):
        # Flatten pixel values
        pixel_values = images.reshape(images.shape[0], -1)

        # Extract Histograms as features
        histograms = np.array([np.histogram(image.flatten(), bins=256, range=[0, 256])[0] for image in images])

        # Extract Local Binary Patterns (LBP) as features
        lbps = np.array([feature.local_binary_pattern(image.astype(int), P=8, R=1, method="uniform").flatten() for image in images])

        # Combine features
        features_combined = np.concatenate((pixel_values, histograms, lbps), axis=1)
        return features_combined
    
    def display_sample_histogram(self):
        histograms = np.array([np.histogram(image.flatten(), bins=256, range=[0, 256])[0] for image in self.training_images])
        first_image_histogram = histograms[0]
        plt.figure(figsize=(8, 4))
        plt.bar(range(256), first_image_histogram, width=1, color='gray')
        plt.title('Histogram for the First Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    def train_model(self):
        # Extract features for training
        training_data_combined = self.extract_features(self.training_images)

        # Train SVM
        self.svm_model = SVC(kernel='linear', C=1.0)
        self.svm_model.fit(training_data_combined, self.training_labels.ravel())

    def validate_model(self):
        # Extract features for validation
        validation_data_combined = self.extract_features(self.validation_images)

        # Validate the model
        validation_predictions = self.svm_model.predict(validation_data_combined)

        # Evaluate on validation set
        val_accuracy = accuracy_score(self.validation_labels, validation_predictions)
        print(f'Validation Accuracy: {100*val_accuracy:.2f}%')

    def test_model(self):
        # Extract features for testing
        testing_data_combined = self.extract_features(self.testing_images)

        # Test the model
        testing_predictons = self.svm_model.predict(testing_data_combined)

        # Evaluate on test set
        test_accuracy = accuracy_score(self.testing_labels, testing_predictons)
        print(f'Test Accuracy: {100*test_accuracy:.2f}%')

        # Classification report for test set
        print("\nClassification Report:")
        print(classification_report(self.testing_labels, testing_predictons))


if __name__ == "__main__":
    classifier = PneumoniaSVMClassifier('../Datasets/pneumoniamnist.npz')
    classifier.train_model()
    classifier.validate_model()
    classifier.test_model()
