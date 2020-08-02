import csv
import tensorflow as tf
import google.protobuf

for j in range(1,22):
    if j<10:
        j=str('0'+str(j))
    else:
        j=str(j)
    with open('pv_'+j+'.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        dataset = []
        for row in spamreader:
            dataset.append(row)
        #print(dataset)

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)

with tf.Session() as session:
    result = session.run(c)
    print(result)

