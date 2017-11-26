import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# saver = tf.train.Saver()
saver = tf.train.import_meta_graph("/home/disooqi/my_model_final.ckpt.meta")
with tf.Session() as sess:
    gg=saver.restore(sess, "/home/disooqi/my_model_final.ckpt")