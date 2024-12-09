import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

x1_data = np.array([73., 93., 89., 96., 73.], dtype=np.float32)
x2_data = np.array([80., 88., 91., 98., 66.], dtype=np.float32)
x3_data = np.array([75., 93., 90., 100., 70.], dtype=np.float32)
y_data = np.array([152., 185., 180., 196., 142.], dtype=np.float32)

w1 = tf.Variable(tf.random.normal([1]), name='weight1')
w2 = tf.Variable(tf.random.normal([1]), name='weight2')
w3 = tf.Variable(tf.random.normal([1]), name='weight3')
b = tf.Variable(tf.random.normal([1]), name='bias')

learning_rate = 1e-5 # 0.00001
optimizer = tf.optimizers.SGD(learning_rate=learning_rate) # 학습율 설정

def cost_funtion():
    hypothesis = x1_data * w1 + x2_data * w2 + x3_data * w3 + b
    return tf.reduce_mean(tf.square(hypothesis - y_data))

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        cost = cost_funtion()
    gradients = tape.gradient(cost, [w1, w2, w3, b])
    optimizer.apply_gradients(zip(gradients, [w1, w2, w3, b]))
    return cost

for step in range(2001):
    cost_val = train_step()
    if step % 10 == 0:
        hypothesis = x1_data * w1 + x2_data * w2 + x3_data * w3 + b
        print(f"Step: {step}, Cost: {cost_val.numpy()}, Prediction: {hypothesis.numpy()}")

'''
1. 기울기 계산 코드
gradients = tape.gradient(cost, [w1, w2, w3, b])
=> 비용함수에 대해 각 변수 w1, w2, w3, b 의 기울기를 구하고, gradients 에 리스트 형태로 저장된다.
ex. gradients = [g_w1, g_w2, g_w3, g_b]

2. 가중치 업데이트 코드
optimizer.apply_gradients(zip(gradients, [w1, w2, w3, b]))
=> optimizer.apply_gradients는 옵티마이저(optimizer)를 사용해 변수의 값을 업데이트한다.
내부 동작
    a. 미리 선언된 변수(w1, w2, w3, b)의 값을 gradients를 사용해 업데이트
    b. 계산된 기울기와 변수 리스트를 쌍으로 묶어, 옵티마이저가 학습률(learning rate)을 적용해 변수를 자동으로 갱신.
'''

'''
Step: 0, Cost: 34547.78125, Prediction: [52.35689  78.30325  68.95964  76.370575 62.863102]
Step: 10, Cost: 21.2744197845459, Prediction: [144.40155 188.90051 177.95049 195.05666 147.21443]
Step: 20, Cost: 20.84674072265625, Prediction: [144.69774 189.22304 178.28572 195.41948 147.45393]
Step: 30, Cost: 20.73576545715332, Prediction: [144.7163  189.21193 178.29214 195.4243  147.43898]
Step: 40, Cost: 20.62540626525879, Prediction: [144.73395 189.19984 178.29759 195.42802 147.42332]
...
...
Step: 1950, Cost: 7.522553443908691, Prediction: [147.36647 187.3992  179.10934 195.97527 145.09412]
Step: 1960, Cost: 7.483508110046387, Prediction: [147.377   187.392   179.1126  195.9774  145.08485]
Step: 1970, Cost: 7.444669246673584, Prediction: [147.3875  187.38483 179.11583 195.97954 145.0756 ]
Step: 1980, Cost: 7.406052589416504, Prediction: [147.39796 187.37766 179.11906 195.98167 145.06636]
Step: 1990, Cost: 7.367635250091553, Prediction: [147.40842 187.37053 179.12228 195.98381 145.05719]
Step: 2000, Cost: 7.329424858093262, Prediction: [147.41882 187.36342 179.12552 195.98593 145.04802]

'''