import numpy as np
import tensorflow as tf
import pandas as pd
data = pd.read_csv('mnist_train.csv')
X = data.drop('label', axis=1).values
y = data['label'].values
with tf.Session() as sess:
    Y = tf.one_hot(y, 10).eval()
hidden  = [256, 256]

def loss(output, y):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output)
    loss = tf.reduce_mean(loss)
    return loss

def training(cost, global_step):
    optimizer = tf.train.GradientDescentOptimizer(
    learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1),
    tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
    tf.float32))
    return accuracy

def layer(inputs, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev = weight_stddev)
    b_init = tf.constant_initializer()
    W = tf.get_variable("W", weight_shape, initializer = w_init)
    b = tf.get_variable("b", bias_shape, initializer = b_init)
    return tf.nn.relu(tf.matmul(inputs, W) + b)

def inference(x):
    with tf.variable_scope('hidden_1'):
        hidden_1 = layer(x, [784, 256], [256])
    with tf.variable_scope('hidden_2'):
        hidden_2 = layer(hidden_1, [256, 256], [256])
    with tf.variable_scope('output'):
        output = layer(hidden_2, [256,10], [10])
    return output

learning_rate = 0.0001
training_epochs = 1000
with tf.Graph().as_default():
    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])
    output = inference(x)
    cost = loss(output, y)
    global_step = tf.Variable(0, name='global_step',
    trainable=False)
    train_op = training(cost, global_step)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, './save/model.ckpt')
        for epoch in range(training_epochs):
            feed_dict = {x : X, y : Y}
            sess.run(train_op, feed_dict=feed_dict)
            cost_curr = sess.run(cost, feed_dict=feed_dict)
            if epoch % 10 == 0:
            	print(cost_curr)
            	saver.save(sess, './save/model.ckpt')
    print("Optimization Finished!")