import tensorflow as tf
from PIL import Image
import numpy as np

dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_training_2018_7_8__H16_M35.gz"
reader = tf.TFRecordReader()

# Names of the features.
bands = ['R', 'G', 'B']
label = 'BIOMS'
featureNames = list(bands)
featureNames.append(label)
# Feature columns
columns = [tf.FixedLenFeature(shape=[241,241], dtype=tf.float32) for k in featureNames]
# Dictionary with names as keys, features as values.
featuresDict = dict(zip(featureNames, columns))
features = {
    'BIOMS': tf.FixedLenFeature((), tf.float32),
    'R': tf.FixedLenFeature((241, 241), tf.float32),
    'G': tf.FixedLenFeature((241, 241), tf.float32),
    'B': tf.FixedLenFeature((241, 241), tf.float32)
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

test1,test2 = tfrecord_input_fn(fileName=dataPath,batchSize=10,shuffle=True)

with tf.Session() as sess:
    a = (sess.run([test1,test2]))

rgb = np.zeros((241,241,3))
R = Image.fromarray(a[0]['R'][0:10][0])
G = Image.fromarray(a[0]['G'][0:10][0])
B = Image.fromarray(a[0]['B'][0:10][0])

img = Image.fromarray(np.uint8(np.dstack((R,G,B))))
img.show()
