import tensorflow as tf

data = tf.placeholder(tf.float32, [None, 6217,40])
target = tf.placeholder(tf.float32, [None, 10])

num_hidden = 2
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 1000
no_of_batches = int(len(train_input)/batch_size)
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print("Epoch - ",str(i))
incorrect = sess.run(error,{data: test_input, target: test_output} )
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()

dataset = pd.read_csv('pv_01.csv')
dat = data.iloc[:,1].values

print(sess.run(model.prediction,{data: dat}))
