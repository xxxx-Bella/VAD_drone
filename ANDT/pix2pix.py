# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Implementation of the Image-to-Image Translation model.
This network represents a port of the following work:
  Image-to-Image Translation with Conditional Adversarial Networks
  Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
  Arxiv, 2017
  https://phillipi.github.io/pix2pix/
A reference implementation written in Lua can be found at:
https://github.com/phillipi/pix2pix/blob/master/models.lua
"""
import collections
import functools

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout
import tf_slim as slim

@slim.add_arg_scope
# layers = tf.contrib.layers
# contrib 是tf-v1中的模块，v2把它集成到其他模块中了：tf.keras.layers.Layer, tf.keras.Model, tf.Module
# tf.contrib.layers 建立神经网络层的高级操作。此包提供了一些操作，它们负责在内部创建以一致方式使用的变量，并为许多常见的机器学习算法提供构建块
def pix2pix_arg_scope():
    """Returns a default argument scope for isola_net. # 默认参数范围
    Returns:
      An arg scope.
    """
    # These parameters come from the online port, which don't necessarily match
    # those in the paper.
    # TODO(nsilberman): confirm these values with Philip.
    instance_norm_params = {
        'center': True,
        'scale': True,
        'epsilon': 0.00001,
    }

    with slim.arg_scope(  # tf.contrib.framework.arg_scope
            [Conv2D, Conv2DTranspose],
            normalizer_fn=layers.instance_norm,
            normalizer_params=instance_norm_params,
            weights_initializer=tf.random_normal_initializer(0, 0.02)) as sc:
        return sc  # 存储给定list_ops集合的默认参数


def upsample(net, num_outputs, kernel_size, method='nn_upsample_conv'):
    """Upsamples the given inputs. # 上采样 放大图像
    Args:
      net: A `Tensor` of size [batch_size, height, width, filters].
      num_outputs: The number of output filters.
      kernel_size: A list of 2 scalars or a 1x2 `Tensor` indicating the scale,
        relative to the inputs, of the output dimensions. For example, if kernel
        size is [2, 3], then the output height and width will be twice and three
        times the input size. # 输出图像的高和宽 放大为原图像的2倍和3倍。即此处的 kernel_size为图像放大倍数
      method: The upsampling method.
    Returns:
      An `Tensor` which was upsampled using the specified method.
    Raises:
      ValueError: if `method` is not recognized.
    """
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]

    if method == 'nn_upsample_conv':  # Upsample通过插值方法完成上采样。所以不需要训练参数
        net = tf.image.resize_nearest_neighbor(net, [kernel_size[0] * height, kernel_size[1] * width]) 
        # resize_nearest_neighbor(images, size)使用最近邻插值调整images为size

        net = Conv2D(filters=num_outputs, kernel_size=[4, 4], padding="SAME", activation=None)(net) 
        # kernel_size=[4, 4]; 默认padding="SAME"：output_shape = ceil( (input_shape - (kernel_size - 1))/stride )

    elif method == 'conv2d_transpose':  # ConvTranspose2d可以理解为卷积的逆过程。所以可以训练参数
        net = Conv2DTranspose(filters=num_outputs, kernel_size=[4, 4], strides=kernel_size, padding="SAME", activation=None)(net) 
        # stride 移动步长；默认padding="SAME"：output_shape = input_shape * stride; padding="VALID"保持最小相交的原则,上下左右均填充kernel_size大小[4, 4]; 
    else:
        raise ValueError('Unknown method: [%s]', method)

    return net


class Block(
    collections.namedtuple('Block', ['num_filters', 'decoder_keep_prob'])):
    """Represents a single block of encoder and decoder processing.
    The Image-to-Image translation paper works a bit differently than the original
    U-Net model. In particular, each block represents a single operation in the
    encoder which is concatenated with the corresponding decoder representation.
    A dropout layer follows the concatenation and convolution of the concatenated
    features.
    """
    # decoder_keep_prob: 元素被保留下来的概率
    pass


def _default_generator_blocks():
    """Returns the default generator block definitions.
    Returns:
      A list of generator blocks.
    """
    return [
        Block(64, 0.5),
        Block(128, 0.5),
        Block(256, 0.5),
        Block(512, 0),
        Block(512, 0),
        Block(512, 0),
        Block(512, 0),
    ]  # 前三个是Unet(generator)中encoder过程处理的块，后四个是decoder处理的块


def pix2pix_generator(net,
                      num_outputs,
                      blocks=None,
                      upsample_method='nn_upsample_conv',
                      is_training=False):  # pylint: disable=unused-argument
    """Defines the network architecture.
    Args:
      net: A `Tensor` of size [batch, height, width, channels]. Note that the
        generator currently requires square inputs (e.g. height=width).
      num_outputs: The number of (per-pixel) outputs.
      blocks: A list of generator blocks or `None` to use the default generator
        definition.
      upsample_method: The method of upsampling images, one of 'nn_upsample_conv'
        or 'conv2d_transpose'
      is_training: Whether or not we're in training or testing mode.
    Returns:
      A `Tensor` representing the model output and a dictionary of model end
        points.
    Raises:
      ValueError: if the input heights do not match their widths.
    """
    end_points = {}

    blocks = blocks or _default_generator_blocks()

    # 获取input image(tensor)的height和width，并检查其合法性（是否相等）
    input_size = net.get_shape().as_list()
    height, width = input_size[1], input_size[2]
    if height != width:
        raise ValueError('The input height must match the input width.')

    input_size[3] = num_outputs  # 每个像素的输出数 (useless? )

    upsample_fn = functools.partial(upsample, method=upsample_method) # partial(func, *arg): 将给定的参数传给func; 相当于固定upsample()的method参数？
    encoder_activations = []

    ###########
    # Encoder #
    ###########
    with tf.variable_scope()('encoder'):  # 创建一个新的variable：'encoder'
        with slim.arg_scope(
                [Conv2D],
                kernel_size=[4, 4],
                stride=2,
                activation_fn=tf.nn.leaky_relu):
            # 分别定义第一层、最后一层与中间层
            for block_id, block in enumerate(blocks):
                # No normalizer for the first encoder layers as per 'Image-to-Image',
                if block_id == 0:
                    # First layer doesn't use normalizer_fn
                    net = Conv2D(net, block.num_filters, normalizer_fn=None)
                elif block_id < len(blocks) - 1:  # 中间层
                    net = Conv2D(net, block.num_filters)
                else:
                    # Last layer doesn't use activation_fn nor normalizer_fn
                    net = Conv2D(
                        net, block.num_filters, activation_fn=None, normalizer_fn=None)

                encoder_activations.append(net)
                end_points['encoder%d' % block_id] = net  # %d: 占位符，填充数字

    ###########
    # Decoder #
    ###########
    reversed_blocks = list(blocks)
    reversed_blocks.reverse()

    with tf.variable_scope()('decoder'):
        # Dropout is used at both train and test time as per 'Image-to-Image',
        with slim.arg_scope([Dropout], is_training=is_training):
            for block_id, block in enumerate(reversed_blocks):
                if block_id > 0:  # 跳层连接：i层直接与n-i层concatenate
                    net = tf.concat([net, encoder_activations[-block_id - 1]], axis=3)

                # The Relu comes BEFORE the upsample op:
                net = tf.nn.relu(net)
                net = upsample_fn(net, block.num_filters, [2, 2])
                if block.decoder_keep_prob > 0:
                    net = Dropout(net, keep_prob=block.decoder_keep_prob)
                end_points['decoder%d' % block_id] = net

    with tf.variable_scope()('output'):
        logits = Conv2D(net, num_outputs, [4, 4], activation_fn=None)  # logits层是指馈入归一化层（如softmax/tanh）的层
        # print(logits)
        # logits = tf.reshape(logits, input_size)

        end_points['logits'] = logits
        end_points['predictions'] = tf.tanh(logits)

    return logits, end_points


# patchGAN
def pix2pix_discriminator(net, num_filters, padding=2, is_training=False):
    """Creates the Image2Image Translation Discriminator.
    Args:
      net: A `Tensor` of size [batch_size, height, width, channels] representing
        the input.
      num_filters: A list of the filters in the discriminator. The length of the
        list determines the number of layers in the discriminator.
      padding: Amount of reflection padding applied before each convolution.
      is_training: Whether or not the model is training or testing.
    Returns:
      A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
      'patches' we're attempting to discriminate and a dictionary of model end
      points.
    """
    del is_training  # 删除变量is_training，解除is_training对1(或0)的引用
    end_points = {}

    num_layers = len(num_filters)

    def padded(net, scope):  # 对Tensor进行映射填充
        if padding:
            spatial_pad = tf.constant(
                [[0, 0], [padding, padding], [padding, padding], [0, 0]], dtype=tf.int32)  # Creates a constant tensor.
            return tf.pad(net, spatial_pad, 'REFLECT')  # tf.pad()填充tensor; mode='REFLECT'映射填充，上下（1维）填充顺序和paddings相反，左右（0维）顺序补齐
        else:
            return net

    # ### architecture
    with slim.arg_scope(
        [slim.conv2d],
        kernel_size=[4, 4],
        padding='valid',
        activation_fn=tf.nn.leaky_relu,
        stride=2
    ):
        # 输入层
        # No normalization on the input layer.
        net = Conv2D(filters=num_filters[0], kernel_size=[4, 4])(padded(net, 'conv0'))
        end_points['conv0'] = net
        # 中间层
        for i in range(1, num_layers - 1):
            net = Conv2D(filters=num_filters[i], kernel_size=[4, 4])(padded(net, 'conv%d' % i))
            end_points['conv%d' % i] = net
        # 最后层
        # Stride 1 on the last layer.
        net = Conv2D(filters=num_filters[-1], kernel_size=[4, 4], strides=1)(padded(net, 'conv%d' % (num_layers - 1)))
        end_points['conv%d' % (num_layers - 1)] = net
        # logits层
        # 1-dim logits, stride 1, no activation, no normalization.
        logits = Conv2D(filters=1, kernel_size=[4, 4], strides=1, activation=None)(padded(net, 'conv%d' % num_layers))
        end_points['logits'] = logits
        end_points['predictions'] = tf.sigmoid(logits)
    return logits, end_points