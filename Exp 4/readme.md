## **Objective**
This experiment aims to classify handwritten digits from the MNIST dataset using a simple neural network. The network will be trained with different activation functions and hidden layer sizes to explore their effect on the accuracy of the model.

# **Model Explanation**
The neural network consists of:

An input layer with 784 units, corresponding to the 28x28 pixel images in the MNIST dataset.

A single hidden layer whose size varies based on the hyperparameter tuning. The activation function used in the hidden layer can be one of ReLU, Sigmoid, or Tanh.

An output layer with 10 units corresponding to the 10 possible digits (0-9) in the MNIST dataset.

The network is trained using the Adam optimizer and the softmax cross-entropy loss function.

Network Architecture:
Input Layer: 784 neurons (28x28 image flattened into a vector)

Hidden Layer: Varies in size (256, 128, or 64 neurons) with a specified activation function (ReLU, Sigmoid, or Tanh)

Output Layer: 10 neurons (one for each digit class, 0-9)

# **Requirements**
TensorFlow 1.x (tensorflow.compat.v1)

NumPy

Matplotlib

Seaborn

Scikit-learn

You can install the required libraries using:

```
pip install tensorflow numpy matplotlib seaborn scikit-learn
```
# **Code Breakdown**
1. # **Data Loading and Preprocessing**
```
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]
```
The MNIST dataset is loaded and split into training and test sets.

The images are flattened into 784-dimensional vectors and normalized to the range [0, 1].

Labels are one-hot encoded.

2. # **Network Parameters and Hyperparameters**
```
input_size = 784
output_size = 10
learning_rate = 0.01
epochs = 50
batch_size = 10
hidden_layer_sizes = [256, 128, 64]
```
The input size corresponds to the flattened 28x28 images.

The output size is 10, representing the digits 0-9.

Hyperparameters like learning rate, number of epochs, and batch size are defined.

Hidden layer sizes are varied to experiment with different network complexities.

3. # **Neural Network Model**
```
def neural_network(x):
    layer1 = activation_func(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    output_layer = tf.add(tf.matmul(layer1, weights['out']), biases['out'])
    return output_layer
```
This function defines the architecture of the neural network, where:

weights['h1'] and biases['b1'] define the parameters for the hidden layer.

The activation function (ReLU, Sigmoid, or Tanh) is applied to the hidden layer.

The final output layer is calculated using the weights and biases for the output layer.

4. # **Loss and Optimization**
```
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
```
The loss is calculated using the softmax cross-entropy loss function.

The Adam optimizer is used to minimize the loss during training.

5. # **Training the Model**
```
for epoch in range(epochs):
    avg_loss = 0
    total_batches = train_images.shape[0] // batch_size
    
    for i in range(total_batches):
        batch_x = train_images[i * batch_size:(i + 1) * batch_size]
        batch_y = train_labels[i * batch_size:(i + 1) * batch_size]
        _, c, acc = sess.run([optimizer, loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
        avg_loss += c
```
The model is trained for the specified number of epochs.

In each epoch, the training data is split into batches, and for each batch, the optimizer updates the weights and biases.

The loss and accuracy are calculated for each batch.

6. # **Testing and Evaluation**
```
test_acc, preds = sess.run([accuracy, predictions], feed_dict={X: test_images, Y: test_labels})
```
After training, the model is evaluated on the test set.

Accuracy is calculated using the test data.

7. # **Visualization**
The loss curve and accuracy curve are plotted for each configuration of the model.

A confusion matrix is generated to visualize the model's performance in classifying digits.

8. # **Results**
```
print("\nHyperparameter Tuning Results:")
for activation, hl_size, acc, time_taken in results:
    print(f"Activation: {activation}, Hidden Layer: {hl_size}, Test Accuracy: {acc:.4f}, Time: {time_taken:.2f}s")
```
After training and testing all configurations, the results are displayed, showing the test accuracy and the execution time for each combination of activation function and hidden layer size.

# **Final Output**
The model provides the test accuracy for each combination of activation function and hidden layer size.

Training progress is visualized through loss and accuracy curves.

A confusion matrix is displayed to assess how well the model classifies each digit.

# **Loss and Accuracy**
During training, the model prints the loss and accuracy for each epoch. The final test accuracy after training is displayed along with the execution time for the entire process.

# **Comments**
The experiment runs multiple configurations of the neural network, each with a different activation function and hidden layer size.

The results are stored in a list and printed at the end, showing the test accuracy for each configuration.

The confusion matrix and accuracy/loss curves provide insights into how well the model is performing during training and testing.
