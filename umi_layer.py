import tensorflow as tf

class UMI_Layer(tf.keras.layers.Layer):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1, **kwargs):
        super(UMI_Layer, self).__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.delta = tf.constant(delta, dtype=tf.float32)

    def build(self, input_shape):
        # Input shape: (batch_size, 4)  [DeltaR, T, V, A]
        if input_shape[-1] != 4:
            raise ValueError("Input tensor must have shape (batch_size, 4)")
        super(UMI_Layer, self).build(input_shape)

    def call(self, inputs):
        delta_r = inputs[:, 0]  # Relative deviation from baseline
        t = inputs[:, 1]       # Trend coefficient
        v = inputs[:, 2]       # Coefficient of variation
        a = inputs[:, 3]       # Anomaly score

        umi = (self.alpha * delta_r + 
               self.beta * t + 
               self.gamma * v + 
               self.delta * a)
        return umi


# Example Usage (in a Keras model):
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # Input: [DeltaR, T, V, A]
    UMI_Layer(),  # The UMI layer
    # ... rest of your model ...
])

# Example Input Data (NumPy array):
import numpy as np
example_data = np.array([
    [0.1, 0.05, 0.2, -0.5],  # Example values for DeltaR, T, V, A
    [0.2, -0.1, 0.1, 0.2],
    [-0.05, 0.15, 0.05, 1.2]
])

# Example usage with TensorFlow tensors
example_tensor = tf.constant(example_data, dtype=tf.float32)

umi_output = model(example_tensor)  # Calculate UMI values
print(umi_output)

# Example to check for alert conditions
critical_threshold = 1.0
alerts = tf.abs(umi_output) > critical_threshold
print(alerts)

# Example to use in training:
# model.compile(optimizer='adam', loss='mse')  # Example compilation
# model.fit(training_data, training_labels, epochs=10)
