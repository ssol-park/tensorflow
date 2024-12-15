import tensorflow as tf
import numpy as np

# 난수 생성 시드 고정 (결과 재현성을 위해)
tf.random.set_seed(777)

# 학습 데이터 (입력 값)
x_data = np.array([[1, 2, 1],
                   [1, 3, 2],
                   [1, 3, 4],
                   [1, 5, 5],
                   [1, 7, 5],
                   [1, 2, 5],
                   [1, 6, 6],
                   [1, 7, 7]], dtype=np.float32)

# 학습 데이터의 레이블 (정답 값, 원-핫 인코딩)
y_data = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]], dtype=np.float32)

# 테스트 데이터 (입력 값)
x_test = np.array([[2, 1, 1],
                   [3, 1, 2],
                   [3, 3, 4]], dtype=np.float32)

# 테스트 데이터의 레이블 (정답 값, 원-핫 인코딩)
y_test = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]], dtype=np.float32)

W = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.random.normal([3]))

def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

def cost_function(X, Y):
    logits = hypothesis(X)
    return tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(logits), axis=1))

optimizer = tf.optimizers.SGD(learning_rate=0.1)

for step in range(201):
    with tf.GradientTape() as tape:
        cost_val = cost_function(x_data, y_data)
    gradient = tape.gradient(cost_val, [W, b])
    optimizer.apply_gradients(zip(gradient, [W, b]))

    if step % 10 == 0:
        print(f"Step: {step}, Cost: {cost_val.numpy()}")

# 테스트 데이터에 대한 모델 예측
logits = hypothesis(x_test)
prediction = tf.argmax(logits, 1)

# 정확도 계산
is_correct = tf.equal(prediction, tf.argmax(y_test, 1)) # 예측 값과 실제 값 비교
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # 정확도 계산

print("Prediction:", prediction.numpy())  # 예측 클래스 출력
print("Accuracy: ", accuracy.numpy())  # 정확도 출력