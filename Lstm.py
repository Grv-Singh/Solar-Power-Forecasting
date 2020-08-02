#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

NUM_EXAMPLES = 10000

train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))

train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    train_output.append([count/20.0])

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print( "test and training data loaded" )

data = tf.placeholder(tf.float32, [None, 20,1]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 1])
num_hidden = 24
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.matmul(last, weight) + bias
cross_entropy = tf.reduce_sum(tf.abs(target - tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
pred_rounded = tf.round( prediction*20 )/20.0
mistakes = tf.not_equal( target, pred_rounded )
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 1000
no_of_batches = int(len(train_input) / batch_size )
epoch = 100
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp = train_input[ptr:ptr+batch_size]
        out = train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print( "Epoch ",str(i) )

incorrect = sess.run(error,{data: test_input, target: test_output})
print( sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}) )
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()