import tensorflow as tf
import numpy as np
from PIL import Image

dataPath = r"E:\School\Graduate_MS\Thesis\tfRecords\tf_NC_training_2018_7_5__H16_M50.gz"
reader = tf.TFRecordReader()

bands = ['VV', 'VH']
label = 'BIOMS'
featureNames = list(bands)
featureNames.append(label)
# Feature columns
columns = [tf.FixedLenFeature(shape=[33,33], dtype=tf.float32) for k in featureNames]
# Dictionary with names as keys, features as values.
featuresDict = dict(zip(featureNames, columns))
print(featuresDict)
features = {
    'BIOMS': tf.FixedLenFeature((), tf.float32),
    'VV': tf.FixedLenFeature((33, 33), tf.float32),
    'VH': tf.FixedLenFeature((33, 33), tf.float32)
}

def parse_tfrecord(example_proto):
    parsed_features = tf.parse_single_example(example_proto, features)
    labels = parsed_features.pop(label)
    return parsed_features, tf.cast(labels, tf.float32)
  # parsed_features = tf.parse_single_example(example_proto, featuresDict)
  # labels = parsed_features.pop(label)
  # return parsed_features, tf.cast(labels, tf.float32)

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

img = Image.fromarray(np.uint8(a[0]['VH'][0:10][0]*255))
# img.show(title="Just a random patch")
