import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

x_data = np.array([[1, 2],
                   [2, 3],
                   [3, 1],
                   [4, 3],
                   [5, 3],
                   [6, 2]], dtype=np.float32)
y_data = np.array([[0],
                   [0],
                   [0],
                   [1],
                   [1],
                   [1]], dtype=np.float32)

W = tf.Variable(tf.random.normal([2, 1], name='weight'))
b = tf.Variable(tf.random.normal([1], name='bias'))


def hypothesis(X):
    return tf.sigmoid(tf.matmul(X, W) + b)

def cost_function(X, Y):
    logits = hypothesis(X)
    return -tf.reduce_mean(Y * tf.math.log(logits) + (1 - Y) * tf.math.log(1 - logits))

# Define optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

def train_step(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_function(X, Y)
    gradients = tape.gradient(cost, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return cost

def accuracy_function(X, Y):
    logits = hypothesis(X)
    predicted = tf.cast(logits > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    return accuracy

for step in range(10001):
    cost_val = train_step(x_data, y_data)
    if step % 200 == 0:
        print(f"Step: {step}, Cost: {cost_val.numpy()}")

# Accuracy
hypothesis_val = hypothesis(x_data)
predicted_val = tf.cast(hypothesis_val > 0.5, dtype=tf.float32)
accuracy_val = accuracy_function(x_data, y_data)

print("\nHypothesis:", hypothesis_val.numpy())
print("Correct (Y):", predicted_val.numpy())
print("Accuracy:", accuracy_val.numpy())
