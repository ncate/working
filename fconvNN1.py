import tensorflow as tf
import geeTFrecords as geetf
import numpy as np


#######################################################################################################################
### USER DEFINED VARIABLES ############################################################################################
#######################################################################################################################
dataPath = r"E:\School\Graduate_MS\0_Thesis\0_Data\tfRecords\tf_NC_all_2018_10_6__H12_M34_FS129_BandsbiocohVVcohVHVVVH.gz"

featureSize = 129
batchSize = 2
epoch_num = 1
lr = 0.0005
batch_per_ep = 1
print_every = 100
# For this method to work the first item in "bands" has to be the label image!
bands = ['bio', 'cohVV', 'cohVH', 'VV', 'VH']
label = 'bio'


#######################################################################################################################
### Build the features dictionary #####################################################################################
#######################################################################################################################
columns = [tf.FixedLenFeature((featureSize-1, featureSize-1, 1), tf.float32) for k in bands]
featuresDict = dict(zip(bands, columns))
featureNames = bands[(-1*(len(bands)-1)):]


# def fconvNN_keras(inputs):
#     initializer = tf.variance_scaling_initializer(scale=0.1)
#     in_shape = list(map(int, NNinputs.shape[-3:]))
#     print(in_shape)
#
#     layers = [
#         tf.layers.Conv2D(input_shape=in_shape, filters=256, kernel_size=5, padding="SAME", activation=tf.nn.relu),
#         tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#         tf.layers.Dropout(0.3),
#         tf.layers.BatchNormalization(),
#         tf.layers.Conv2D(filters=256, padding="SAME", kernel_size=3, activation=tf.nn.relu),
#         tf.layers.Dropout(0.3),
#         tf.layers.BatchNormalization(),
#         tf.layers.Conv2D(filters=128, padding="SAME", kernel_size=3, activation=tf.nn.relu),
#         tf.layers.Dropout(0.25),
#         tf.layers.BatchNormalization(),
#         tf.layers.Conv2D(filters=96, padding="SAME", kernel_size=3, activation=tf.nn.relu),
#         tf.layers.Dropout(0.25),
#         tf.layers.BatchNormalization(),
#         tf.layers.Conv2D(filters=64, padding="SAME", kernel_size=3, activation=tf.nn.relu),
#         tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#         tf.layers.Dropout(0.25),
#         tf.layers.BatchNormalization(),
#         tf.layers.Flatten(input_shape=in_shape),
#         tf.layers.Dense(units=64, kernel_initializer=initializer, activation=tf.nn.tanh),
#         tf.layers.Dropout(0.25),
#         tf.layers.Dense(units=10, kernel_initializer=initializer, activation=None)
#     ]
#     model = tf.keras.Sequential(layers)
#     return model(inputs)

def fconvNN(inputs):
    net = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], strides=1, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, padding='SAME')
    # print(net.shape)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, padding='SAME')
    # print(net.shape)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, padding='SAME')
    # print(net.shape)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[2, 2], strides=2, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    # print()


    net = tf.layers.conv2d_transpose(inputs=net, filters=64, kernel_size=[3, 3], strides=2, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=[3, 3], strides=2, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.conv2d_transpose(inputs=net, filters=8, kernel_size=[3, 3], strides=2, padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=[5, 5], strides=2, padding='SAME',  activation=tf.nn.relu)
    # print(net.shape)
    return net


    ####################################################################################################################
  ### PREPPING THE SESSION ###########################################################################################
####################################################################################################################
trainImgs, trainLabels = geetf.get_fcon_geeTFrecord(fileName=dataPath,
                                               featureDictionary=featuresDict,
                                               labelKey=label,
                                               numEpochs=epoch_num,
                                               shuffle=True,
                                               batchSize=batchSize)
trainLabels = tf.image.resize_image_with_crop_or_pad(trainLabels, featureSize - 1, featureSize - 1)
for i in list(trainImgs.keys()):
    trainImgs[i] = tf.image.resize_image_with_crop_or_pad(trainImgs[i], featureSize-1, featureSize-1)

testImgs, testLabels = geetf.get_fcon_geeTFrecord(fileName=dataPath,
                                               featureDictionary=featuresDict,
                                               labelKey=label,
                                               numEpochs=epoch_num,
                                               shuffle=True,
                                               batchSize=batchSize)
testLabels = tf.image.resize_image_with_crop_or_pad(testLabels, featureSize - 1, featureSize - 1)
for i in list(testImgs.keys()):
    testImgs[i] = tf.image.resize_image_with_crop_or_pad(testImgs[i], featureSize-1, featureSize-1)


NNinputs = tf.placeholder(tf.float32, (batchSize, featureSize - 1, featureSize - 1, len(featureNames)))
NNoutputs = fconvNN(NNinputs)

# initialize the network
init = tf.global_variables_initializer()


def trainNN(net_fn, num_epochs):
    scores = net_fn(NNinputs)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=trainLabels, logits=scores)
    loss = tf.losses.huber_loss(predictions=NNoutputs, labels=trainLabels)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(trainImgs['VV'])
        # _, c = sess.run([train_op, loss], feed_dict={NNinputs: a})
        # print('Epoch: {} - cost = {:.5f}'.format((ep + 1), c))


trainNN(net_fn=fconvNN, num_epochs=10)