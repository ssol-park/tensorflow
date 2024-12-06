import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 렌더링 설정
import matplotlib.pyplot as plt

# 데이터
X = np.array([1, 2, 3], dtype=np.float32)
Y = np.array([1, 2, 3], dtype=np.float32)

# 1. 학습 파라미터 W 정의 (랜덤 초기값)
W = tf.Variable(tf.random.normal([1]), name='weight')

# 2. 옵티마이저 정의 (SGD 사용)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 3. 비용 함수 정의
def cost_function():
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))  # MSE 계산

# 4. 학습 단계 정의
@tf.function
def train_step():
    with tf.GradientTape() as tape:  # GradientTape로 기울기 계산
        cost = cost_function()
    gradients = tape.gradient(cost, [W])  # W에 대한 기울기 계산
    optimizer.apply_gradients(zip(gradients, [W]))  # W 업데이트
    return cost

# 5. 학습 과정 출력 및 기록
W_val = []  # W 값을 저장할 리스트
cost_val = []  # 비용 함수 값을 저장할 리스트

# 학습 반복
for step in range(100):  # 100번 반복 학습
    cost = train_step()  # 한 스텝 학습
    W_val.append(W.numpy()[0])  # W 값을 리스트에 저장
    cost_val.append(cost.numpy())  # 비용 함수 값을 리스트에 저장

    # 10단계마다 출력
    if step % 10 == 0:
        print(f"Step {step}, Cost: {cost.numpy()}, W: {W.numpy()}")

# 6. 비용 함수 그래프 그리기
plt.plot(W_val, cost_val)
plt.xlabel("W")
plt.ylabel("Cost")
plt.title("Cost vs W")
plt.savefig("cost_vs_w.png")  # 이미지를 파일로 저장
print("학습 완료 및 그래프 저장 완료.")
