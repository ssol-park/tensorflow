import tensorflow as tf
import numpy as np

tf.random.set_seed(777)  # for reproducibility

# 데이터 로드 및 준비
data = np.loadtxt('../data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = data[:, :-1]
y_data = data[:, [-1]]

# `tf.data.Dataset`으로 데이터 배치 처리
batch_size = 10
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)

# 모델 정의
class LinearRegressionModel(tf.keras.Model):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.W = tf.Variable(tf.random.normal([3, 1]), name='weight')
        self.b = tf.Variable(tf.random.normal([1]), name='bias')

    def call(self, X):
        return tf.matmul(X, self.W) + self.b

# 모델 및 손실 함수 정의
model = LinearRegressionModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)

def cost_function(X, Y):
    hypothesis = model(X)
    return tf.reduce_mean(tf.square(hypothesis - Y))

# 학습 루프
for step in range(2001):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            cost_val = cost_function(x_batch, y_batch)
        gradients = tape.gradient(cost_val, [model.W, model.b])
        optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    if step % 10 == 0:
        print(f"Step {step}, Cost: {cost_val.numpy()}")

# 예측
print("Your score will be ", model(tf.constant([[100, 70, 101]], dtype=tf.float32)).numpy())
print("Other scores will be ", model(tf.constant([[60, 70, 110], [90, 100, 80]], dtype=tf.float32)).numpy())
