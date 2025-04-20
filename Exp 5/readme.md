# **CNN Model for Fashion MNIST Classification**
# **Objective**
The objective of this project is to design and evaluate a Convolutional Neural Network (CNN) for classifying images from the Fashion MNIST dataset. The project explores the impact of different hyperparameters, such as filter size, regularization, batch size, and optimizer, on the performance of the model. The evaluation metrics include loss and accuracy, which are tracked over the training epochs.

# **Model Explanation**
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture is as follows:

Convolutional Layer (Conv2D): The model starts with a 2D convolutional layer with 32 filters and a kernel size of 3x3 (default). The ReLU activation function is used to introduce non-linearity.

Max Pooling Layer (MaxPooling2D): A max pooling layer follows each convolutional layer to downsample the feature maps, reducing dimensionality while retaining important features.

Second Convolutional Layer (Conv2D): A second convolutional layer with 64 filters and a kernel size of 3x3 (default) is used to extract higher-level features.

Flatten Layer: The 2D feature maps are flattened into a 1D vector to feed into fully connected layers.

Dense Layer: A fully connected layer with 128 neurons and ReLU activation is used to learn complex features.

Output Layer (Dense): The final output layer has 10 units (one for each class in the Fashion MNIST dataset) with a softmax activation function to output class probabilities.

# **Hyperparameters**
Filter Size: The kernel size for the convolutional layers, with values tested being 3x3 and 5x5.

Regularization: L2 regularization is applied to prevent overfitting, with the option to use or omit it.

Batch Size: Batch sizes tested include 32 and 64.

Optimizer: The model uses two optimizers, Adam and Stochastic Gradient Descent (SGD), for comparison.

# **Requirements**
To run this code, the following Python libraries are required:

tensorflow (Keras is included within TensorFlow)

matplotlib

numpy

You can install them using pip:

```
pip install tensorflow matplotlib numpy
```
# **Code Breakdown**
1. Importing Libraries
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import numpy as np
```
tensorflow is used for building and training the CNN.

keras provides the high-level API for neural networks.

matplotlib is used for plotting the loss and accuracy curves.

2. Enabling GPU Acceleration
```
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
   ```
This code enables GPU acceleration if available, which can speed up model training.

4. Loading and Preprocessing the Fashion MNIST Dataset
```
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```
The Fashion MNIST dataset is loaded, containing grayscale images of 10 different fashion items.

The pixel values are normalized to the range [0, 1] by dividing by 255.

The data is reshaped to have 1 channel (grayscale) for each image.

4. Creating the CNN Model
```
def create_cnn(filter_size=3, reg=None, optimizer='adam'):
    model = keras.Sequential([
        layers.Conv2D(32, (filter_size, filter_size), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=reg),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (filter_size, filter_size), activation='relu', kernel_regularizer=reg),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
   ```
This function defines the CNN model using the Keras Sequential API.

The model consists of two convolutional layers, followed by max-pooling layers, and a dense layer for classification.

The optimizer, regularizer, and filter size are customizable.

5. Training and Plotting the Model's Performance
```
def train_and_plot(model, batch_size=32, title="Model Performance"):
    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
    final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'{title} - Final Loss: {final_loss:.4f}, Final Accuracy: {final_accuracy * 100:.2f}%')
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title + ' - Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title + ' - Loss')

    plt.show()
   ```
This function trains the model for 10 epochs, evaluates the model's final loss and accuracy, and plots the accuracy and loss curves for both training and validation sets.

7. Testing with Different Hyperparameters
The code then tests the model with different hyperparameters by calling train_and_plot with various combinations of filter size, regularization, batch size, and optimizer.


# Test different filter sizes
```
for size in [3, 5]:
    model = create_cnn(filter_size=size)
    train_and_plot(model, title=f'Filter Size {size}')
```
# Test different regularization techniques
```
for reg in [None, regularizers.l2(0.001)]:
    model = create_cnn(reg=reg)
    train_and_plot(model, title=f'Regularization {"L2" if reg else "None"}')
```
# Test different batch sizes
```
for batch in [32, 64]:
    model = create_cnn()
    train_and_plot(model, batch_size=batch, title=f'Batch Size {batch}')
```
# Test different optimizers
```
for opt in ['adam', 'sgd']:
    model = create_cnn(optimizer=opt)
    train_and_plot(model, title=f'Optimizer {opt.upper()}')
```
# **Final Output**
The final output consists of the following:

Loss and Accuracy Values: After each experiment, the model's final loss and accuracy are printed.

Training and Validation Curves: Accuracy and loss curves for both training and validation sets are plotted for each hyperparameter combination tested.

# **Loss and Accuracy**
Loss: The loss function used is sparse categorical cross-entropy, which is appropriate for multi-class classification tasks.

Accuracy: Accuracy is tracked as the primary metric to evaluate model performance.

# **Comments**
GPU Acceleration: Enabling GPU acceleration improves training speed if a compatible GPU is available.

Hyperparameter Tuning: The script allows testing different combinations of hyperparameters, helping to understand their impact on the model's performance.

