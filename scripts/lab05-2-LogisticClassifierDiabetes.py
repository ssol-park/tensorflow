import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(777)

# 데이터 로드
xy = np.loadtxt('../data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]  # 입력 특징 데이터
y_data = xy[:, [-1]]  # 레이블 데이터 (출력)

# 1. 입력 데이터 정규화
# 입력 데이터의 크기를 일정한 범위로 조정하여 학습 안정성을 높이고
# 특성(feature) 간 크기 차이가 학습에 영향을 주지 않도록 한다.
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 2. 학습 데이터와 테스트 데이터로 분리
# 학습 데이터로 모델을 "공부"시키고, 테스트 데이터로 "시험"을 보아
# 모델의 일반화 성능(새로운 데이터에 대한 성능)을 평가한다.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=777)

# 3. 로지스틱 회귀 모델 정의
class LogisticRegressionModel(tf.keras.Model):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.W = tf.Variable(tf.random.normal([8, 1]), name='weight')
        self.b = tf.Variable(tf.random.normal([1]), name='bias')

    def call(self, inputs):
        return tf.sigmoid(tf.matmul(inputs, self.W) + self.b)

# 모델 초기화
model = LogisticRegressionModel()

# 비용 함수 (크로스 엔트로피)
# 예측값(hypothesis)과 실제값(y) 간의 차이를 계산.
# 예측이 정확할수록 비용이 작아지고, 틀릴수록 비용이 커진다.
def cost_function(X, Y):
    logits = model(X)
    return -tf.reduce_mean(Y * tf.math.log(logits) + (1 - Y) * tf.math.log(1 - logits))

# 옵티마이저 정의 (경사 하강법)
optimizer = tf.optimizers.SGD(learning_rate=0.01)


# 비용 함수에 따라 가중치와 절편을 업데이트하여 학습을 진행한다
def train_step(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_function(X, Y)
    gradients = tape.gradient(cost, [model.W, model.b])  # 비용에 따른 기울기 계산
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))  # 기울기를 사용하여 파라미터 업데이트
    return cost


# 예측값과 실제값이 얼마나 일치하는지 계산한다
def compute_accuracy(X, Y):
    logits = model(X)
    predicted = tf.cast(logits > 0.5, dtype=tf.float32)  # 0.5 기준으로 True/False 결정
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))  # 정확도 계산
    return accuracy

# 학습 루프
for step in range(10001):
    cost_val = train_step(x_train, y_train)
    if step % 200 == 0:
        train_accuracy = compute_accuracy(x_train, y_train)
        test_accuracy = compute_accuracy(x_test, y_test)
        print(f"Step: {step}, Cost: {cost_val.numpy()}, Train Accuracy: {train_accuracy.numpy()}, Test Accuracy: {test_accuracy.numpy()}")

final_hypothesis = model(x_data)
final_predicted = tf.cast(final_hypothesis > 0.5, dtype=tf.float32)
final_accuracy = compute_accuracy(x_test, y_test)

print("\nHypothesis:", final_hypothesis.numpy())
print("Predicted:", final_predicted.numpy())
print("Final Test Accuracy:", final_accuracy.numpy())
