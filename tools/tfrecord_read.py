#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
import argparse
import logging
import h5py

logging.basicConfig(format='%(asctime)s %(threadName)s [%(levelname)s] [line:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level = logging.DEBUG
                    #level = logging.WARN
                    )


parser = argparse.ArgumentParser(description='Read TFRecords')

parser.add_argument('--filepath', type=str, default='/gfs/atlastts/StandFemale_22K/tfrecords/mcep_cut_normal/test/007934.tfrecords', help='file path to the TFRecords.')
parser.add_argument('--metapath', type=str, default='/gfs/atlastts/StandFemale_22K/tfrecords/mcep_cut_normal/meta/mcep_cut/mean-std-meta.hdf5', help='file path to the hdf5.')


FLAGS = parser.parse_args()

# h5py
f = h5py.File(FLAGS.metapath, 'r')
logging.info('Keys: {}'.format(f.keys()))

for k in f.keys():
    logging.info('{} keys values: {}'.format(k, f[k][:]))

frame = -1
# tfrecord
record_iter = tf.python_io.tf_record_iterator(FLAGS.filepath)
for r in record_iter:
        print('start...')
	example_proto = tf.train.Example()
	example_proto.ParseFromString(r)

	data_encoding_id = example_proto.features.feature['encoding'].int64_list.value[0]

        height = example_proto.features.feature['height'].int64_list.value[0]
        width = example_proto.features.feature['width'].int64_list.value[0]
        channel = example_proto.features.feature['depth'].int64_list.value[0]
        print(height, width, channel)

	data_raw_image = example_proto.features.feature['image_raw'].bytes_list.value[0]
	data_label = example_proto.features.feature['label'].bytes_list.value[0]

        image_raw = np.fromstring(data_raw_image, np.float32)
        image_raw = np.reshape(image_raw, [height, width, channel])
        logging.info(image_raw.shape)
        logging.info(image_raw[:, frame, :].flatten())
        logging.info('mean {}, std {}'.format(np.mean(image_raw), np.std(image_raw)))

        label = np.fromstring(data_label, np.float32)
        label = np.reshape(label, [height, width, channel])
        logging.info(label.shape)
        logging.info(label[:, frame, :].flatten())

        logging.info('mean {}, std {}'.format(np.mean(label), np.std(label)))

