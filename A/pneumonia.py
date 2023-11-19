import numpy as np
import matplotlib.pyplot as plt

pneumonia_dataset = np.load('./Datasets/pneumoniamnist.npz')
print(pneumonia_dataset.files)

training_images = pneumonia_dataset['train_images']
validation_images = pneumonia_dataset['val_images']
testing_images = pneumonia_dataset['test_images']

training_labels = pneumonia_dataset['train_labels']
validation_labels = pneumonia_dataset['val_labels']
testing_labels = pneumonia_dataset['test_labels']

plt.imshow(training_images[0], cmap='gray')
plt.title('Sample image')
plt.show()

unique_values, counts = np.unique(training_labels.flatten(), return_counts=True)

# Print the individual labels and their count
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")