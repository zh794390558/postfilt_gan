#!/usr/bin/env python

'''
GV(global variance)
'''
import os
import sys
import numpy as np
import argparse
import logging
import h5py

logging.basicConfig(format='%(asctime)s %(threadName)s [%(levelname)s] [line:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level = logging.DEBUG
                    #level = logging.WARN
                    )


parser = argparse.ArgumentParser(description='Read TFRecords')

parser.add_argument('--metapath', type=str, default='/gfs/atlastts/StandFemale_22K/tfrecords/mcep_cut_normal/meta/mcep_cut/mean-std-meta.hdf5', help='file path to the hdf5.')
parser.add_argument('--mceppath', type=str, default=None, help='file path to the SYN mcep.')

FLAGS = parser.parse_args()

# h5py
f = h5py.File(FLAGS.metapath, 'r')
logging.info('Keys: {}'.format(f.keys()))

for k in f.keys():
    logging.info('{} keys values: {}'.format(k, f[k][:]))

syn_mean=f['gen_mean'][:]
syn_std=f['gen_std'][:]
mean = f['nature_mean'][:]
std = f['nature_std'][:]


data = np.fromfile(FLAGS.mceppath, np.float32)

data = np.reshape(data, (-1, 41))
logging.info('mcep shape({}):\n{}'.format(data.shape, data))

data = (data - syn_mean) / syn_std
data = data * std + mean

logging.info('mcep GV shape({}):\n{}'.format(data.shape, data))

data.astype(np.float32).tofile(FLAGS.mceppath)

sys.exit(0)

