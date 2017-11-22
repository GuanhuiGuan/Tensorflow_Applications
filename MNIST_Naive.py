import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# variables
pixels = 784
tags = 10
x = tf.placeholder(tf.float32, [None, pixels])
W = tf.Variable(tf.zeros([pixels, tags]))
b = tf.Variable(tf.zeros([tags]))
# output softmax Wx+b
# the whole thing is transposed in code
y = tf.nn.softmax(tf.matmul(x, W) + b)

# training set up: minimize loss
# here cross entropy == loss
# y_: correct labels
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# train loop
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for t in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(t)

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prob = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print(prob)