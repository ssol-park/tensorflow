import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

x_data = np.array([[1, 2, 1, 1],
                   [2, 1, 3, 2],
                   [3, 1, 3, 4],
                   [4, 1, 5, 5],
                   [1, 7, 5, 5],
                   [1, 2, 5, 6],
                   [1, 6, 6, 6],
                   [1, 7, 7, 7]], dtype=np.float32)

y_data = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]], dtype=np.float32)
nb_classes = 3
n_features = x_data.shape[1] # 4 / x_data.shape (8, 4)

class SoftmaxClassifier(tf.keras.Model):
    def __init__(self, n_features, nb_classes):
        super(SoftmaxClassifier, self).__init__()
        self.W = tf.Variable(tf.random.normal([n_features, nb_classes]), name='weight')
        self.b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

    def call(self, inputs):
        logits = tf.matmul(inputs, self.W) + self.b
        return tf.nn.softmax(logits)

model = SoftmaxClassifier(n_features, nb_classes)

def cost_function(X, Y):
    logits = model(X)
    return -tf.reduce_mean(tf.reduce_sum(Y * tf.math.log(logits), axis=1))

optimizer = tf.optimizers.SGD(learning_rate=0.1)

def train_step(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_function(X, Y)
    gradients = tape.gradient(cost, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))
    return cost

# 예측 함수
def predict(X):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    logits = model(X)

    # argmax: 배열의 행 or 열 을 기준으로 최댓값의 인덱스를 반환한다.
    # axis: 0:열 기준, 1:행 기준
    return tf.argmax(logits, axis=1)

for step in range(2001):
    cost_val = train_step(x_data, y_data)
    if step % 200 == 0:
        print(f"Step: {step}, Cost: {cost_val.numpy()}")

print('--------------')
a = model(np.array([[1, 11, 7, 9]], dtype=np.float32))
print(a.numpy(), predict([[1, 11, 7, 9]]).numpy())

print('--------------')
b = model(np.array([[1, 3, 4, 3]], dtype=np.float32))
print(b.numpy(), predict([[1, 3, 4, 3]]).numpy())

print('--------------')
c = model(np.array([[1, 1, 0, 1]], dtype=np.float32))
print(c.numpy(), predict([[1, 1, 0, 1]]).numpy())

print('--------------')
all_data = np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]], dtype=np.float32)
all_predictions = model(all_data)
print(all_predictions.numpy(), predict(all_data).numpy())