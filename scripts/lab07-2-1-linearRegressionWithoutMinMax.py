import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

# 데이터 준비
# 해당 데이터는 크기와 범위가 매우 다르다.
# - 첫 번째 열은 주가처럼 800~830 사이의 값.
# - 세 번째 열은 거래량처럼 100만 단위의 값.
# 데이터 크기 차이 때문에 모델이 가중치를 학습할 때 제대로 된 계산을 못하므로 손실 값(cost)이 매우 커지거나 발산해서 'nan'이 발생할 가능성이 크다
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1].astype(np.float32)  # 입력 데이터
y_data = xy[:, [-1]].astype(np.float32)  # 출력 데이터

class LinearModel(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([4, 1]), name='weight')
        self.b = tf.Variable(tf.random.normal([1]), name='bias')

    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b

def cost_function(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

learning_rate = 1e-5
optimizer = tf.optimizers.SGD(learning_rate= learning_rate)

model = LinearModel()

for step in range(101):
    with tf.GradientTape() as tape:
        y_pred = model(x_data)

        cost_val = cost_function(y_pred, y_data)

    gradients = tape.gradient(cost_val, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    if step % 10 == 0:
        print(f"Step: {step}, Cost: {cost_val.numpy()}")
        print(f"Prediction:\n{y_pred.numpy()}")