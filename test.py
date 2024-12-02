# test.py
import tensorflow as tf

# 간단한 Tensor 생성 및 연산
a = tf.constant([1, 2, 3], dtype=tf.float32)
b = tf.constant([4, 5, 6], dtype=tf.float32)
c = tf.add(a, b)

print("Tensor a:", a.numpy())
print("Tensor b:", b.numpy())
print("Tensor a + b:", c.numpy())