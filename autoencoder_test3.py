import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

dataPath = r"E:\School\Graduate_MS\0_Thesis\0_Data\tfRecords\tf_NC_all_2018_8_2__H18_M48_FS33_BandsVVVH.gz" # RADAR - S1
# dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_all_2018_7_11__H22_M31_FS33_BandsB11B8B4B3B2.gz"   # OPTICAL - S2
# dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_all_2018_7_11__H22_M24_FS33_BandsRGBN.gz"   # OPTICAL - NAIP
reader = tf.TFRecordReader()
featureSize = 33
batchSize = 10  # Number of samples in each batch
epoch_num = 20    # Number of epochs to train the network
lr = 0.0001       # Learning rate
batch_per_ep = 40
bands = ['VV', 'VH']
label = 'BIOMS'

writeLossCSV = True
showReconstructedImages = True
showLossGraph = True

# Build the features dictionary
labelDict = {label: tf.FixedLenFeature((), tf.float32)}
columns = [tf.FixedLenFeature((featureSize, featureSize), tf.float32) for k in bands]
featuresDict = labelDict
featuresDict.update(dict(zip(bands, columns)))

def parse_tfrecord(example_proto):
    parsed_features = tf.parse_single_example(example_proto, featuresDict)
    labels = parsed_features.pop(label)
    return parsed_features, tf.cast(labels, tf.float32)

def tfrecord_input_fn(fileName,
                      numEpochs=None,
                      shuffle=True,
                      batchSize=None):
  dataset = tf.data.TFRecordDataset(fileName, compression_type='GZIP')
  # Map the parsing function over the dataset
  dataset = dataset.map(parse_tfrecord)
  # Shuffle, batch, and repeat.
  if shuffle:
    dataset = dataset.shuffle(buffer_size=batchSize * 10)
  dataset = dataset.batch(batchSize)
  dataset = dataset.repeat(numEpochs)
  # Make a one-shot iterator.
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

def autoencoder(inputs):
    # newWidth = (Width - filterSize + 2*Padding) / Stride + 1

    # encoder
    # First convolutional layer: 32 x 32 x 1  ->  32 x 32 x 32
    # First max-pooling layer: 32 x 32 x 32  ->  16 x 16 x 32
    #
    # Second convolutional layer: 16 x 16 x 32  ->  16 x 16 x 16
    # Second max-pooling layer: 16 x 16 x 16  ->  8 x 8 x 16
    #
    # Third convolutional layer: 8 x 8 x 16  ->  8 x 8 x 8
    # Third max-pooling layer: 8 x 8 x 16  ->  2 x 2 x 8
    #
    # Fourth convolutional layer: 1 x 1 x 16
    net = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], strides=1, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[5,5], strides=2, padding='SAME')

    net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[5, 5], strides=1, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[5, 5], strides=2, padding='SAME')

    net = tf.layers.conv2d(inputs=net, filters=8, kernel_size=[5, 5], strides=1, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[5, 5], strides=4, padding='SAME')

    # Addition of one extra convolutional layer
    net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[2, 2], strides=1, padding='valid', activation=tf.nn.relu)

    # decoder
    # added: 1 x 1 x 15  ->  2 x 2 x 8
    # 2 x 2 x 8  ->  8 x 8 x 16
    # 8 x 8 x 16  ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = tf.layers.conv2d_transpose(inputs=net, filters=8, kernel_size=[2, 2], strides=1, padding='valid', activation=tf.nn.relu)

    net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=[5, 5], strides=4, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=[5, 5], strides=2, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(inputs=net, filters=2, kernel_size=[5, 5], strides=2, padding='SAME', activation=tf.nn.relu)
    return net


ae_inputs = tf.placeholder(tf.float32, (batchSize, featureSize-1, featureSize-1, len(bands)))
ae_outputs = autoencoder(ae_inputs)  # runs the autoencoder

# loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
loss = tf.losses.huber_loss(predictions=ae_outputs, labels=ae_inputs)  # calculate the huber loss
# loss = tf.losses.cosine_distance(predictions=ae_outputs, labels=ae_inputs)  # calculate the cosine distance loss

train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

lossList = []
epList = []

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            train1, train2 = tfrecord_input_fn(fileName=dataPath, shuffle=True, batchSize=batchSize)
            a = sess.run([train1, train2])
            toFeed = np.zeros((batchSize, featureSize-1, featureSize-1, len(bands)))
            for i in range(len(a[0]["VH"])):
                vh = np.delete(a[0]["VH"][i], featureSize-1, 0)
                vh = np.delete(vh, featureSize-1, 1)
                vh = np.resize(vh, (featureSize-1, featureSize-1, 1))

                vv = np.delete(a[0]["VV"][i], featureSize-1, 0)
                vv = np.delete(vv, featureSize-1, 1)
                vv = np.resize(vv, (featureSize-1, featureSize-1, 1))
                temp = np.dstack((vv, vh))
                toFeed[i] = temp

            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: toFeed})
            lossList.append(c)
            epList.append(ep)
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
    # test the trained network
    test1, test2 = tfrecord_input_fn(fileName=dataPath, shuffle=True, batchSize=batchSize)
    a = sess.run([test1, test2])
    toFeed = np.zeros((batchSize, featureSize-1, featureSize-1, len(bands)))
    for i in range(len(a[0]["VH"])):
        vh = np.delete(a[0]["VH"][i], featureSize - 1, 0)
        vh = np.delete(vh, featureSize - 1, 1)
        vh = np.resize(vh, (featureSize - 1, featureSize - 1, 1))

        vv = np.delete(a[0]["VV"][i], featureSize - 1, 0)
        vv = np.delete(vv, featureSize - 1, 1)
        vv = np.resize(vv, (featureSize - 1, featureSize - 1, 1))
        temp = np.dstack((vv, vh))
        toFeed[i] = temp

    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: toFeed})[0]

    if writeLossCSV:
        # Make a csv of the loss for each batch. Do with it what you will
        now = datetime.datetime.now()
        f = open("loss/Loss_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute) + ".csv", 'w')
        writer = csv.writer(f)
        for i in range(len(lossList)):
            row = [(epList[i] + 1), lossList[i]]
            writer.writerow(row)
        f.close()

    if showReconstructedImages:
        # Display the input images and outputs from the autoencoder.
        plt.figure(1)
        for i in range(9):
            plt.title('Reconstructed')
            plt.subplot(3, 3, i+1)
            plt.imshow(recon_img[i, ..., 0], cmap='gray')

        plt.figure(2)
        for i in range(9):
            plt.title('Input')
            plt.subplot(3, 3, i+1)
            plt.imshow(toFeed[i, ..., 0], cmap='gray')
        plt.show()

    if showLossGraph:
        avgLoss_perEp = []
        for e in range(epoch_num):
            i = e*batch_per_ep
            epLoss = []
            for b in range(batch_per_ep):
                epLoss.append(lossList[i+b])
            avgLoss_perEp.append((sum(epLoss)/batch_per_ep))
        plt.figure(3)
        plt.title('Average Loss Value by Epoch')
        plt.plot(list(range(epoch_num)),avgLoss_perEp, '.r-')
        plt.show()
