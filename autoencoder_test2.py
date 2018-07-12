import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

# dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_all_2018_7_9__H20_M38_FS33_BandsVVVH.gz" # RADAR - S1
# dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_all_2018_7_11__H22_M31_FS33_BandsB11B8B4B3B2.gz"   # OPTICAL - S2
dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_all_2018_7_11__H22_M24_FS33_BandsRGBN.gz"   # OPTICAL - NAIP
reader = tf.TFRecordReader()
featureSize = 33
batchSize = 10  # Number of samples in each batch
epoch_num = 15    # Number of epochs to train the network
lr = 0.001        # Learning rate
batch_per_ep = 55

# bands = ['VV', 'VH']
label = 'BIOMS'
# featureNames = list(bands)
# featureNames.append(label)
# # Feature columns
# columns = [tf.FixedLenFeature(shape=[featureSize,featureSize], dtype=tf.float32) for k in featureNames]
# # Dictionary with names as keys, features as values.
# featuresDict = dict(zip(featureNames, columns))
# # print(featuresDict)
# features = {
#     'BIOMS': tf.FixedLenFeature((), tf.float32),
#     'VV': tf.FixedLenFeature((featureSize, featureSize), tf.float32),
#     'VH': tf.FixedLenFeature((featureSize, featureSize), tf.float32)
# }
features = {
    'BIOMS': tf.FixedLenFeature((), tf.float32),
    'R': tf.FixedLenFeature((featureSize, featureSize), tf.float32),
    'G': tf.FixedLenFeature((featureSize, featureSize), tf.float32),
    'B': tf.FixedLenFeature((featureSize, featureSize), tf.float32)
}

def parse_tfrecord(example_proto):
    parsed_features = tf.parse_single_example(example_proto, features)
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
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x  32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    # net = lays.conv2d(inputs, 64, [5, 5], stride=2, padding='SAME')
    net = tf.layers.conv2d(inputs, 32, [5, 5], strides=2, padding='SAME')
    net = tf.layers.conv2d(net, 16, [5, 5], strides=2, padding='SAME')
    net = tf.layers.conv2d(net, 8, [5, 5], strides=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = tf.layers.conv2d_transpose(net, 16, [5, 5], strides=4, padding='SAME')
    net = tf.layers.conv2d_transpose(net, 32, [5, 5], strides=2, padding='SAME')
    net = tf.layers.conv2d_transpose(net, 1, [5, 5], strides=2, padding='SAME', activation=tf.nn.relu)
    return net


ae_inputs = tf.placeholder(tf.float32, (batchSize, 32, 32, 1))
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
# calculate the loss and optimize the network
# loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
loss = tf.losses.huber_loss(predictions = ae_outputs, labels = ae_inputs)  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
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
            toFeed = np.zeros((batchSize, featureSize-1, featureSize-1, 1))
            for i in range(len(a[0]["G"])):
                temp = np.delete(a[0]["G"][i], featureSize-1, 0)
                temp = np.delete(temp, featureSize-1, 1)
                temp = np.resize(temp, (featureSize-1, featureSize-1, 1))
                toFeed[i] = temp
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: toFeed})
            lossList.append(c)
            epList.append(ep)
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
    # test the trained network
    test1, test2 = tfrecord_input_fn(fileName=dataPath, shuffle=True, batchSize=batchSize)
    a = sess.run([test1, test2])
    toFeed = np.zeros((batchSize, featureSize-1, featureSize-1, 1))
    for i in range(len(a[0]["G"])):
        temp = np.delete(a[0]["G"][i], featureSize - 1, 0)
        temp = np.delete(temp, featureSize - 1, 1)
        temp = np.resize(temp, (featureSize-1, featureSize-1, 1))
        toFeed[i] = temp
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: toFeed})[0]

    # Make a csv of the loss for each batch. Do with it what you will
    now = datetime.datetime.now()
    f = open("loss/Loss_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute) + ".csv", 'w')
    writer = csv.writer(f)
    for i in range(len(lossList)):
        row = [(epList[i] + 1), lossList[i]]
        writer.writerow(row)
    f.close()

    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(toFeed[i, ..., 0], cmap='gray')
    plt.show()

