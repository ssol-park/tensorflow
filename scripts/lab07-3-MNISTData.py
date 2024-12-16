import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.datasets import mnist

tf.random.set_seed(777)

# MNIST 이미지는 (28, 28) 크기의 2D 배열로 표현 되며, 이를 펼치면 (784,) 크기의 벡터가 된다. => 입력 데이터의 열 수는 항상 784 이다.
# MNIST 데이터는 0부터 9까지의 숫자 이미지를 분류하는 문제 이므로 10개의 클래스(숫자 0 ~ 9)가 존재한다 => 출력 데이터의 열 수는 항상 10 이다.
# MNIST 데이터 로드 및 로컬 저장
try:
    # 로컬에 저장된 데이터가 있다면 사용
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='/app/data/mnist.npz')
    print("Load MNIST data from local.")
except Exception as e:
    print("## Failed to load MNIST data locally, attempt to  download MNIST data.")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    np.savez('/app/data/mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("## Download MNIST data and save it locally.")

# 데이터 전처리 (정규화 및 차원 변경)
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# 레이블 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

nb_classes = 10  # 클래스 개수

# 모델 정의
class SoftmaxClassifier(tf.keras.Model):
    def __init__(self, input_dim, nb_classes):
        super(SoftmaxClassifier, self).__init__()
        self.W = tf.Variable(tf.random.normal([input_dim, nb_classes]), name='weight')
        self.b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

    def call(self, inputs):
        logits = tf.matmul(inputs, self.W) + self.b
        return tf.nn.softmax(logits)

model = SoftmaxClassifier(784, nb_classes)

# 비용 함수 (크로스 엔트로피)
def cost_function(X, Y):
    logits = model(X)
    return -tf.reduce_mean(tf.reduce_sum(Y * tf.math.log(logits), axis=1))

# 옵티마이저 정의
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# 학습 함수
def train_step(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_function(X, Y)
    gradients = tape.gradient(cost, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))
    return cost

# 정확도 계산 함수
def compute_accuracy(X, Y):
    logits = model(X)
    predicted = tf.argmax(logits, axis=1)
    actual = tf.argmax(Y, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, actual), tf.float32))
    return accuracy

# 학습 설정
num_epochs = 15 # 학습을 몇 번 반복할지 결정 (1 epoch = 전체 데이터셋을 한 번 학습)
batch_size = 100 # 한 번 학습에 사용할 데이터의 개수, 메모리 절약과 학습 안정성을 위함
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# 학습 루프
for epoch in range(num_epochs):
    avg_cost = 0
    for batch_x, batch_y in train_dataset:
        cost_val = train_step(batch_x, batch_y)
        avg_cost += cost_val / len(x_train) * batch_size
    print(f"Epoch: {epoch + 1}, Cost: {avg_cost.numpy()}")

print("Learning finished")

# 테스트 정확도 출력
test_accuracy = compute_accuracy(x_test, y_test)
print("Test Accuracy: ", test_accuracy.numpy())

# 랜덤 테스트 및 이미지 저장
r = random.randint(0, x_test.shape[0] - 1)
prediction = tf.argmax(model(x_test[r : r + 1]), axis=1).numpy()
actual = np.argmax(y_test[r : r + 1], axis=1)

print(f"Label: {actual}")
print(f"Prediction: {prediction}")

# 이미지 저장
plt.imshow(x_test[r : r + 1].reshape(28, 28), cmap="Greys", interpolation="nearest")
plt.title(f"Label: {actual[0]}, Prediction: {prediction[0]}")
plt.savefig(f"/app/data/mnist_test_{r}.png")
print(f"Image saved as /app/data/mnist_test_{r}.png")
