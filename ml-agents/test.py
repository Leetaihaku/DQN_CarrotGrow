import tensorflow.compat.v1 as tf
import numpy as np
import time

with tf.device("/gpu:0"):
    x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    tf.disable_v2_behavior()
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    W = tf.Variable(tf.random_normal([2,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([2]), name='Bias1')
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
    b2 = tf.Variable(tf.zeros([1]), name='Bias2')
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# 여기서부터는 같은 코드를 사용합니다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 시작 시간 측정
start_time = time.time()
start = time.gmtime(start_time)
print("훈련 시작 : %d시 %d분 %d초"%(start.tm_hour, start.tm_min, start.tm_sec))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
                print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))  # W로 하나 W2로 하나 차이 없음.

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

# 끝난 시간 측정
end_time = time.time()
end = time.gmtime(end_time)
print("훈련 끝 : %d시 %d분 %d초"%(end.tm_hour, end.tm_min, end.tm_sec))

# 소요 시간 측정
end_start = end_time - start_time
end_start = time.gmtime(end_start)
print("소요시간 : %d시 %d분 %d초"%(end_start.tm_hour, end_start.tm_min, end_start.tm_sec))