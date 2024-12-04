import tensorflow as tf

# 1.Build graph using TF operations
x_train = tf.constant([1, 2, 3], dtype=tf.float32)
y_train = tf.constant([1, 2, 3], dtype=tf.float32)

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

#  최적화 함수 정의
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 1. 가설 정의
def hypothesis(x):
    return  x * W + b

# 2. cost function 정의
def cost_function():
    return tf.reduce_mean(tf.square(hypothesis(x_train) - y_train))

# 3.학습 단계 정의 (gradient descent)
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        cost = cost_function()
    gradients = tape.gradient(cost, [W, b])           # 가중치와 바이어스에 대한 gradient 계산
    optimizer.apply_gradients(zip(gradients, [W, b])) # 최적화 적용
    return cost

for step in range(20):
    cost = train_step()
    if step % 5 == 0:
        print(f"Step {step}, Cost: {cost.numpy()}, W: {W.numpy()}, b: {b.numpy()}")

print("Result: ", hypothesis(x_train).numpy())