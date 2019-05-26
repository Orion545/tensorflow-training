import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IN_NUM = 784;
OUT_NUM = 10;
mnist = input_data.read_data_sets("MINST_data/",one_hot = True);

x = tf.placeholder(float,[None,IN_NUM]);
W = tf.Variable(tf.zeros([IN_NUM,OUT_NUM]));
b = tf.Variable(tf.zeros([OUT_NUM]));

y = tf.nn.softmax(tf.matmul(x,W) + b);
z = tf.placeholder(float,[None,10]);

cross_entropy = -tf.reduce_sum(z * tf.log(y));
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy);

init = tf.initialize_all_variables();

sess = tf.Session();
sess.run(init);

for _ in range(1,2000):
	mnist_in,mnist_out = mnist.train.next_batch(100);
	sess.run(train,feed_dict={x:mnist_in,z:mnist_out});

correct = tf.equal(tf.arg_max(y,1),tf.arg_max(z,1));
accuracy = tf.reduce_mean(tf.cast(correct,float));

mnist_in = mnist.test.images;
mnist_out = mnist.test.labels;
print(sess.run(accuracy,feed_dict={x:mnist_in,z:mnist_out}));

