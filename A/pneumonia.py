import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import time

# Load the dataset
pneumonia_dataset = np.load('./Datasets/pneumoniamnist.npz')

# Images
training_images = pneumonia_dataset['train_images']
validation_images = pneumonia_dataset['val_images']
testing_images = pneumonia_dataset['test_images']

# Labels
training_labels = pneumonia_dataset['train_labels']
validation_labels = pneumonia_dataset['val_labels']
testing_labels = pneumonia_dataset['test_labels']

# Normalize pixel values to be between 0 and 1
training_images = training_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0
testing_images = testing_images.astype('float32') / 255.0

# # Display sample image (remove this before submission)
# plt.imshow(training_images[0], cmap='gray')
# plt.title('Sample image')
# plt.show()

# Print the individual labels and their count
unique_values, counts = np.unique(training_labels.flatten(), return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")


# Function to build the CNN model and customize the parameters
def build_model(image_height, image_width, number_of_channels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, number_of_channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    return model

def build_model2(image_height, image_width, number_of_channels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, number_of_channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    return model


image_height, image_width = training_images[0].shape
number_of_channels = 1 # Because it is a grayscale image

model = build_model(image_height, image_width, number_of_channels)
# model = build_model2(image_height, image_width, number_of_channels)


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=10, batch_size=32, validation_data=(validation_images, validation_labels))

time.sleep(5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(testing_images, testing_labels)
print(f'Test accuracy: {test_acc}')


