import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
hidden_layer = 625
output_size = 10
training = tf.placeholder_with_default(True, shape=())
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
           self.training = tf.placeholder(tf.bool)

	   self.keep_prob = tf.placeholder(tf.float32)

# input place holders
	   self.X = tf.placeholder(tf.float32, [None, 784])
	   X_img = tf.reshape(self.X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
	   self.Y = tf.placeholder(tf.float32, [None, 10])

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
	   self.logits = tf.matmul(hidden_L4 , hidden_w5) + hidden_b5

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        
#        self.cost = (ready_cost + 0.001*regularization)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()

models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

'''
0 Accuracy: 0.9933
1 Accuracy: 0.9946
2 Accuracy: 0.9934
3 Accuracy: 0.9935
4 Accuracy: 0.9935
5 Accuracy: 0.9949
6 Accuracy: 0.9941

Ensemble accuracy: 0.9952
'''
