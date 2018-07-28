
# coding: utf-8


import cv2 # opencv
import tensorflow as tf #tensorflow
import glob # to read files
import numpy as np
import os
from random import shuffle


dirImage = './jaffe/'
filenames = []
labels = []
filenames += glob.glob(dirImage+"/*"+".tiff")
for i in filenames:
    labels.append(''.join(i.strip().split('.')[2:3])[:-1])
#print(labels)
#print("Processing " + str(len(filenames)) + " images")
#print(filenames)


def get_sift(img):
    #img = denormalize_image(img)
    img = cv2.imread(img)
    #img = np.array(img*255, dtype="uint8")
    gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,None)
    #cv2.imshow('result', img)
    #cv2.waitKey(0)
    #cv2.destroyWindow("result")
    return img



images = []
for file in filenames:
    #img = np.asarray(cv2.imread(file, 0))
    img = get_sift(file)
    images.append(img)
#images = np.asarray(images)
#print(images)


emotion = ['NE', 'HA', 'SA', 'SU', 'AN', 'DI', 'FE' ]
for i in range(7):
    for j in range(len(labels)):
        if labels[j] == emotion[i]:
            labels[j] = i
labels_count = len(emotion)
#print(labels)
TRAINING_SIZE = 170
VALIDATION_SIZE = len(images) - TRAINING_SIZE
temp = np.array([images, labels])
temp = temp.transpose()
shuffle(temp)
images = list(temp[:, 0])
images = np.stack(images, axis = 0) 
labels = list(temp[:, 1])
labels = [int(i) for i in labels]
train_images = images[:TRAINING_SIZE, :, : ] # 170, 256, 256
test_images = images[TRAINING_SIZE:, :, : ]
print(type(train_images))


#label_file = open('temp_labels.txt');
#label_data = label_file.read()
#labels = []
#for i in xrange(len(label_data)):
   # if (label_data[i]!='\n'):
       # labels.append(float(label_data[i]))
train_labels = np.asarray(labels[:TRAINING_SIZE], dtype=int)
test_labels = np.asarray(labels[170:], dtype=int)
#print(train_labels.shape[0])


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # print(type(np.int_(index_offset + labels_dense.ravel())))
    labels_one_hot.flat[np.int_(index_offset + labels_dense.ravel())] = 1
    return labels_one_hot


train_labels = dense_to_one_hot(train_labels.ravel(), labels_count)
# labels = labels.astype(np.uint8)
test_labels = dense_to_one_hot(test_labels.ravel(), labels_count)
#print(train_labels.shape)


train_images = train_images.reshape(TRAINING_SIZE, -1)
test_images = test_images.reshape(VALIDATION_SIZE, -1)
print(train_images.shape)


train_image_pixels = train_images.shape[1]
#print('Flat pixel values is %d'%(train_image_pixels))


train_image_width = np.int_(np.ceil(np.sqrt(train_image_pixels/3.0)))
train_image_height = np.int_(np.ceil(np.sqrt(train_image_pixels/3.0)))




# # Build TF CNN model


def new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup input filter state
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    
    # initialise weights and bias
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup convolutional layer 
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    
    # applying relu non-linear activation
    out_layer = tf.nn.relu(out_layer)
    
    # performing max pooling
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    
    return out_layer

# input & output of NN

# images
x = tf.placeholder(tf.float32, [None, train_image_width * train_image_height*3.0])
# dynamically reshaping input
x_shaped = tf.reshape(x, [-1, train_image_width, train_image_height, 3])
# labels
y = tf.placeholder(tf.float32, [None, labels_count])
keep_prob = tf.placeholder(tf.float32)

# creating sparse layers of CNN
conv1 = tf.layers.conv2d(x_shaped, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name='conv1')
bn1 = tf.layers.batch_normalization(conv1, training=True, name='bn1')
pool1 = tf.layers.max_pooling2d(bn1, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool1')
conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name='conv2')
bn2 = tf.layers.batch_normalization(conv2, training=True, name='bn2')
pool2 = tf.layers.max_pooling2d(bn2, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool2')
conv3 = tf.layers.conv2d(pool1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name='conv3')
bn3 = tf.layers.batch_normalization(conv3, training=True, name='bn3')
pool3 = tf.layers.max_pooling2d(bn3, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool3')
conv4 = tf.layers.conv2d(pool1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name='conv4')
bn4 = tf.layers.batch_normalization(conv4, training=True, name='bn4')
pool4 = tf.layers.max_pooling2d(bn4, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool4')

flatten_layer = tf.contrib.layers.flatten(pool4, 'flatten_layer')
weights = tf.get_variable(shape=[flatten_layer.shape[-1], labels_count], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1), name='fc_weights')
biases = tf.get_variable(shape=[labels_count], dtype=tf.float32,initializer=tf.constant_initializer(0.0), name='fc_biases')

logit_output = tf.nn.bias_add(tf.matmul(flatten_layer, weights), biases, name='logit_output')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit_output))
y_ = tf.nn.softmax(logit_output)
pred_label = tf.argmax(logit_output, 1)
label = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_label, label), tf.float32))
'''
layer1 = new_conv_layer(x_shaped, 3, 32, [5, 5], [2, 2], name='layer1')
bn1 = tf.layers.batch_normalization(layer1, training=True, name='bn1')
pool1 = tf.layers.max_pooling2d(bn1, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool1')
layer2 = new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
bn2 = tf.layers.batch_normalization(conv2, training=True, name='bn2')
layer3 = new_conv_layer(layer2, 64, 128, [5, 5], [2, 2], name='layer3')
layer4 = new_conv_layer(layer3, 128, 256, [5, 5], [2, 2], name='layer4')
flattened = tf.reshape(layer4, [-1, 16 * 16 * 256])
#flattened = tf.reshape(layer1, [-1, 32 * 32 * 512])


# calculating dense layers of CNN
wd1 = tf.Variable(tf.truncated_normal([16 * 16 * 256, 500], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([500], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# dropout for reducing overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(dense_layer1, keep_prob)

# another layer for softmax calculation and readout
wd2 = tf.Variable(tf.truncated_normal([500, labels_count], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([labels_count], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)


keep_prob = tf.placeholder(tf.float32)
wd2 = tf.Variable(tf.truncated_normal([32 * 32 * 512, labels_count], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([labels_count], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(flattened, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)
'''
# cross entropy cost function
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))


# # Training CNN


# set to 3000 iterations 
epochs = 100
DROPOUT = 0.5
batch_size = 16

# settings
learning_rate = 1e-3

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# adding optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# defining accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setting up the initialisation operator
init_op = tf.global_variables_initializer()

# recording variable to store accuracy
tf.summary.scalar('accuracy', accuracy)
saver = tf.train.Saver()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs')

with tf.Session() as sess:
    # initialising variables
    sess.run(init_op)
    total_batch = int(len(train_labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        batch_index = 0
        for j in range(total_batch):
            train_batch_images = train_images[batch_index:(batch_index+batch_size), :]
            train_batch_labels = train_labels[batch_index:(batch_index+batch_size), :]
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x:train_batch_images, y:train_batch_labels, keep_prob:DROPOUT})
            avg_cost += c
            batch_index += batch_size
        train_acc = sess.run(accuracy, feed_dict={x:train_images, y:train_labels})
        test_acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "train accuracy: {:.3f}".format(train_acc), "test accuracy: {:.3f}".format(test_acc))
        save_path = './model/jaffe-model.ckpt'
        saver.save(sess, save_path)
        summary = sess.run(merged, feed_dict={x: test_images, y: test_labels})
        writer.add_summary(summary, epoch)
    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: test_images, y: test_labels}))

