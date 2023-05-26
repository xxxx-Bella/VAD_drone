# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import unet
import pix2pix

from flownet2.src.flowlib import flow_to_image
from flownet2.src.flownet_sd.flownet_sd import FlowNetSD
from flownet2.src.training_schedules import LONG_SCHEDULE
from flownet2.src.net import Mode

# slim = tf.contrib.slim
import tf_slim as slim


def generator(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=3):
    return unet.unet(inputs, layers, features_root, filter_size, pool_size, output_channel)


def discriminator(inputs, num_filers=(128, 256, 512, 512)):
    logits, end_points = pix2pix.pix2pix_discriminator(inputs, num_filers)
    return logits, end_points['predictions']  # logits is a tensor [batch_size, N, N, 1]


def flownet(input_a, input_b, height, width, reuse=None):
    net = FlowNetSD(mode=Mode.TEST)
    # train preds flow
    input_a = (input_a + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    input_b = (input_b + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    # input size is 384 x 512
    input_a = tf.image.resize(input_a, [height, width])
    input_b = tf.image.resize(input_b, [height, width])
    flows = net.model(
        inputs={'input_a': input_a, 'input_b': input_b},
        training_schedule=LONG_SCHEDULE,
        trainable=False, reuse=reuse
    )
    return flows['flow']  # 一个Tensor 什么值？


def initialize_flownet(sess, checkpoint):
    flownet_vars = slim.get_variables_to_restore(include=['FlowNetSD'])
    flownet_saver = tf.train.Saver(flownet_vars)
    print('FlownetSD restore from {}'.format(checkpoint))
    flownet_saver.restore(sess, checkpoint)
