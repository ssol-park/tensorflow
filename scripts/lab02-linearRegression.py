import tensorflow as tf

# 1.Build graph using TF operations
x_train = tf.constant([1, 2, 3], dtype=tf.float32)
y_train = tf.constant([1, 2, 3], dtype=tf.float32)

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis(가설)
hypothesis = x_train * W + b

print("Hypothesis: ", hypothesis)

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # 평균 제곱 오차 (MSE)
print("Cost: ", cost.numpy())

