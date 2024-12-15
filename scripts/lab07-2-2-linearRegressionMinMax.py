import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

# 데이터 준비
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])


# Min-Max 스케일링 함수 (Normalization 의 기법 중 하나)
def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    # 0으로 나누는 문제를 방지하기 위해 작은 값 추가
    return numerator / (denominator + 1e-7)



# Min-Max 스케일링 적용
xy = min_max_scaler(xy)

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


''' def min_max_scaler(data):
1. 각 열에서 최소값을 뺌
모든 값을 최소값 기준으로 0부터 시작하도록 변환
numerator = data - np.min(data, 0)

예시:
data:       [[828.659973, 833.450012, 908100], 
             [823.02002, 828.070007, 1828100], 
             [819.929993, 824.400024, 1438100],
             [816, 820.958984, 1008100],
             [819.359985, 823, 1188100],
             [819, 823, 1198100],
             [811.700012, 815.25, 1098100],
             [809.51001, 816.659973, 1398100]]
최소값(min): [809.51001, 815.25, 908100, 804.539978, 809.559998]
numerator:  [[ 19.149963,  18.200012,       0],
             [ 13.51001,   12.820007,  919000],
             [ 10.419983,   9.150024,  530000],
             [  6.48999,    5.708984,  100000],
             [  9.849975,   7.75,      280000],
             [  9.48999,    7.75,      290000],
             [  2.189995,   0,        190000],
             [  0,          1.409973,  490000]]

2. 각 열에서 최대값과 최소값의 차이를 계산
데이터가 최대-최소 범위 안으로 조정됨
denominator = np.max(data, 0) - np.min(data, 0)

예시:
최대값(max): [828.659973, 833.450012, 1828100, 828.349976, 831.659973]
최소값(min): [809.51001, 815.25, 908100, 804.539978, 809.559998]
denominator: [ 19.149963,  18.200012,  919000,   23.810997,   22.099975]

3. 각 값에 대해 Min-Max 스케일링 계산
(각 값 - 최소값) / (최대값 - 최소값)
scaled_data = numerator / (denominator + 1e-7)

예시:
numerator:  [[ 19.149963,  18.200012,       0],
             [ 13.51001,   12.820007,  919000],
             [ 10.419983,   9.150024,  530000],
             [  6.48999,    5.708984,  100000],
             [  9.849975,   7.75,      280000],
             [  9.48999,    7.75,      290000],
             [  2.189995,   0,        190000],
             [  0,          1.409973,  490000]]
denominator: [ 19.149963,  18.200012,  919000,   23.810997,   22.099975]
scaled_data: [[1.0, 1.0, 0.0],
              [0.705, 0.705, 1.0],
              [0.544, 0.502, 0.576],
              [0.339, 0.314, 0.109],
              [0.514, 0.426, 0.304],
              [0.496, 0.426, 0.315],
              [0.114, 0.0,   0.206],
              [0.0,   0.077, 0.533]]
'''