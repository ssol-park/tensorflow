import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)
node3 = tf.add(node1, node2)
print("node1: ", node1, "node2: ", node2, "node3: ", node3)

print("1. ===============================")
print("node1:", node1.numpy(), "node2:", node2.numpy(), "node3:", node3.numpy())  # 3.0 # 4.0 # 7.0


print("2. ===============================")
@tf.function
def compute_sum(a, b):
    return a + b

sum = compute_sum(node1, node2)
print("sum.numpy():", sum.numpy())

print("3. ===============================")
@tf.function
def add_tensors(x, y):
    return x + y

x = tf.constant([1, 1])
y = tf.constant([2, 1])

add = add_tensors(x, y)
print("add.numpy()", add.numpy())
