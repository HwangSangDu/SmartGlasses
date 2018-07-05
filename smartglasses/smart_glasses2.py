import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
hidden_layer = 625
output_size = 10
training = tf.placeholder_with_default(True, shape=())

keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])


filter_shape = [ 3, 3, 1, 32] 	   
W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
b = tf.Variable(tf.constant(0.1, shape=[32]), name="b1")  # Do equals output_size with bias

conv = tf.nn.conv2d(
                    X_img,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")

bn = tf.contrib.layers.batch_norm(
	            conv,
		    data_format = 'NHWC',
                    center=True,
                    scale=True,
                    is_training = training,
                    scope = 'cnn3d-batch_norm')

h = tf.nn.relu(tf.nn.bias_add(bn, b), name="relu")


filter_shape2 = [ 3, 3, 32, 32]
W11 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W11")
b11 = tf.Variable(tf.constant(0.1, shape=[32]), name="b11")

conv11 = tf.nn.conv2d(
                    h,
                    W11,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv11")

bn2 = tf.contrib.layers.batch_norm(
	            conv11,
		    data_format = 'NHWC',
       	            center=True,
       	            scale=True,
       	            is_training = training,
       	            scope = 'cnn3d-batch_norm2')

h11 = tf.nn.relu(tf.nn.bias_add(bn2, b11), name="relu11")

pooled = tf.nn.max_pool(
                    h11,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="SAME",
                    name="pool")

dropout = tf.nn.dropout(pooled , 0.7)

filter_shape2 = [ 3, 3, 32, 64]
W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W2")
b2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b2")
conv2 = tf.nn.conv2d(
                    dropout,
              	    W2,
              	    strides=[1, 1, 1, 1],
              	    padding="SAME",
               	    name="conv2")

bn3 = tf.contrib.layers.batch_norm(
	            conv2,
		    data_format = 'NHWC',
                    center=True,
                    scale=True,
                    is_training = training,
                    scope = 'cnn3d-batch_norm3')

h2 = tf.nn.relu(tf.nn.bias_add(bn3, b2), name="relu2")

filter_shape21 = [ 3, 3, 64, 64]
W21 = tf.Variable(tf.truncated_normal(filter_shape21, stddev=0.1), name="W21")
b21 = tf.Variable(tf.constant(0.1, shape=[64]), name="b21")
conv21 = tf.nn.conv2d(
                    h2,
                    W21,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv11")
	    
bn4 = tf.contrib.layers.batch_norm(
	            conv21,
		    data_format = 'NHWC',
                    center=True,
                    scale=True,
                    is_training = training,
                    scope = 'cnn3d-batch_norm4')

h21 = tf.nn.relu(tf.nn.bias_add(bn4, b21), name="relu21")
pooled2 = tf.nn.max_pool(
                    h21,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="SAME",
                    name="pool2")

dropout2 = tf.nn.dropout(pooled2 , 0.7)

filter_shape3 = [ 3, 3, 64, 128]
W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W3")
b3 = tf.Variable(tf.constant(0.1, shape=[128]), name="b3")
conv3 = tf.nn.conv2d(
                    dropout2,
              	    W3,
              	    strides=[1, 1, 1, 1],
              	    padding="SAME",
               	    name="conv3")

bn5 = tf.contrib.layers.batch_norm(
	            conv3,
		    data_format = 'NHWC',
                    center=True,
                    scale=True,
                    is_training = training,
                    scope = 'cnn3d-batch_norm5')

h3 = tf.nn.relu(tf.nn.bias_add(bn5, b3), name="relu3")

filter_shape31 = [ 3, 3, 128, 128]
W31 = tf.Variable(tf.truncated_normal(filter_shape31, stddev=0.1), name="W31")
b31 = tf.Variable(tf.constant(0.1, shape=[128]), name="b31")
conv31 = tf.nn.conv2d(
                    h3,
                    W31,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv31")
	    
bn6 = tf.contrib.layers.batch_norm(
	            conv31,
		    data_format = 'NHWC',
                    center=True,
                    scale=True,
                    is_training = training,
                    scope = 'cnn3d-batch_norm6')


h31 = tf.nn.relu(tf.nn.bias_add(bn6, b31), name="relu31")
pooled3 = tf.nn.max_pool(
                    h31,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="SAME",
                    name="pool3")
dropout3 = tf.nn.dropout(pooled3 , 0.7)

L3_flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])

hidden_w1 = tf.get_variable("w4" , shape=[128 * 4 * 4, hidden_layer] , initializer=tf.contrib.layers.xavier_initializer())
hidden_b1 = tf.Variable(tf.random_normal([hidden_layer]))
hidden_L1 = tf.nn.relu(tf.matmul(L3_flat , hidden_w1) + hidden_b1)
#    hidden_L1 = tf.nn.dropout(hidden_L1 , 0.7)

hidden_w2 = tf.get_variable("w5" , shape=[hidden_layer, hidden_layer] , initializer=tf.contrib.layers.xavier_initializer())
hidden_b2 = tf.Variable(tf.random_normal([hidden_layer]))
hidden_L2 = tf.nn.relu(tf.matmul(hidden_L1 , hidden_w2) + hidden_b2)
#    hidden_L2 = tf.nn.dropout(hidden_L2 , 0.7)

hidden_w3 = tf.get_variable("w6" , shape=[hidden_layer, hidden_layer] , initializer=tf.contrib.layers.xavier_initializer())
hidden_b3 = tf.Variable(tf.random_normal([hidden_layer]))
hidden_L3 = tf.nn.relu(tf.matmul(hidden_L2 , hidden_w3) + hidden_b3)
	#    hidden_L3 = tf.nn.dropout(hidden_L3 , 0.7)

hidden_w4 = tf.get_variable("w7" , shape=[hidden_layer, hidden_layer] , initializer=tf.contrib.layers.xavier_initializer())
hidden_b4 = tf.Variable(tf.random_normal([hidden_layer]))
hidden_L4 = tf.matmul(hidden_L3 , hidden_w4) + hidden_b4

hidden_w5 = tf.get_variable("w8" , shape=[hidden_layer, output_size] , initializer=tf.contrib.layers.xavier_initializer())
hidden_b5 = tf.Variable(tf.random_normal([output_size]))
logits = tf.matmul(hidden_L4 , hidden_w5) + hidden_b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy

# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))


