# **Objective:**
To develop a Recurrent Neural Network (RNN) model for time series prediction using synthetic sine wave data, demonstrating sequence learning with PyTorch.

# **Description of Model:**
The model is a simple RNN-based regressor designed to predict the next value in a sine wave sequence:

Architecture:
Input Size: 1
Hidden Size: 50
Output Size: 1

Layers: 1-layer RNN followed by a fully connected (linear) layer

# **Purpose:**
To learn the temporal pattern of sine waves and predict the next point in a sequence.


# **Description of Code:**

Data Generation:
Generates sine wave data over the range [0, 100] with 1000 points using np.sin.

# **Preprocessing:**
Scales data to range [0, 1] using MinMaxScaler.

Creates overlapping sequences of length seq_length (default = 20) for training.

Model Definition:
A class RNNPredictor inheriting from nn.Module with:

An RNN layer for sequence modeling
A linear layer for output projection

# **Training:**
Uses Mean Squared Error (MSE) loss.
Optimized with Adam optimizer (learning rate = 0.01).
Trained over 100 epochs, printing loss every 10 epochs.

# **Evaluation:**

The model is evaluated on a held-out test set (20% of the data).
Test predictions are plotted against true values using matplotlib.

# **Performance Evaluation:**
Training Loss: Gradually decreases over epochs, indicating successful learning.
Test Loss: Final MSE loss is printed after evaluation.
Visual Output: A plot shows predicted vs. actual sine wave values, which visually confirms the model's ability to follow the trend of the sequence.


# **Comments:**

The model performs well for synthetic data, but for real-world noisy data, more advanced models (e.g., LSTM, GRU) might be beneficial.

Limitations:
Might not work well on real-world noisy data
Basic RNNs can forget older info; using LSTM or GRU could be better.
No early stopping, so it might train too much.
No hyperparameter tuning — values like learning rate and sequence length are fixed.


