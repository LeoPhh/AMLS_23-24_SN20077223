import numpy as np
import matplotlib.pyplot as plt

pathmnist_dataset = np.load('./Datasets/pathmnist.npz')
print(pathmnist_dataset.files)

training_images = pathmnist_dataset['train_images']
validation_images = pathmnist_dataset['val_images']
testing_images = pathmnist_dataset['test_images']

training_labels = pathmnist_dataset['train_labels']
validation_labels = pathmnist_dataset['val_labels']
testing_labels = pathmnist_dataset['test_labels']

plt.imshow(training_images[2], cmap='gray')
plt.title('Sample image')
plt.show()

unique_values, counts = np.unique(training_labels.flatten(), return_counts=True)

# Print the individual labels and their count
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")