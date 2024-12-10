import tensorflow as tf
import numpy as np

# 랜덤 시드 설정 (재현 가능성 보장)
tf.random.set_seed(777)

# 데이터 로드
xy = np.loadtxt('../data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# 데이터 확인
# print(x_data, "\nx_data shape:", x_data.shape)
# print(y_data, "\ny_data shape:", y_data.shape)

# 모델 정의
class LinearRegressionModel(tf.keras.Model):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.W = tf.Variable(tf.random.normal([3, 1]), name='weight')
        self.b = tf.Variable(tf.random.normal([1]), name='bias')

    def call(self, X):
        return tf.matmul(X, self.W) + self.b

# 인스턴스 생성
model = LinearRegressionModel()

def cost_function(X, Y):
    hypothesis = model(X)
    return tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)

for step in range(2001):
    with tf.GradientTape() as tape:
        cost_val = cost_function(x_data, y_data)
    gradients = tape.gradient(cost_val, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    if step % 10 == 0:
        print(step, "Cost: ", cost_val.numpy())

# 예측
print("Your score will be ", model(tf.constant([[100, 70, 101]], dtype=tf.float32)).numpy())
print("Others score will be ", model(tf.constant([[60, 70, 110], [90, 100, 80]], dtype=tf.float32)).numpy())