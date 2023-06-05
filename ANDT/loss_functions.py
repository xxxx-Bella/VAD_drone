import tensorflow as tf
import numpy as np


def flow_loss(gen_flows, gt_flows): # optical flow loss
    print(gen_flows['flow'])
    return tf.reduce_mean(tf.abs(gen_flows['flow'] - gt_flows['flow']))


def intensity_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames. #求两者的Lp范数

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    return tf.reduce_mean(tf.abs((gen_frames - gt_frames) ** l_num))

# 设计计算梯度的网络
def gradient_loss(gen_frames, gt_frames, alpha):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale. # shape=[None,256,256,3]
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised. # 每个梯度项的幂

    @return: The GDL loss.
    """
    # calculate the loss for each scale
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.

    channels = gen_frames.get_shape().as_list()[-1] # gen_frames的最后一维为通道数（3）
    ## 分别构造两个维度的卷积核，用于求两个维度的梯度
    pos = tf.constant(np.identity(channels), dtype=tf.float32)   # np.identity(n)返回一个 n×n 单位矩阵; positive 
    neg = -1 * pos # negative; shape=(3, 3)
    # tf.stack([neg, pos]).shape=(2, 3, 3)
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # expand_dims(t,0)使t增加了一个维度,shape=(1, 2, 3, 3)
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # shape=(2, 1, 3, 3)
    '''
    filter_x:                      filter_y:
    [[  [[-1., -0., -0.],          [ [[[ 1.,  0.,  0.],
         [-0., -1., -0.],              [ 0.,  1.,  0.],
         [-0., -0., -1.]],             [ 0.,  0.,  1.]]],

        [[ 1.,  0.,  0.],            [[[-1., -0., -0.],
         [ 0.,  1.,  0.],              [-0., -1., -0.],
         [ 0.,  0.,  1.]] ]]           [-0., -0., -1.]]] ]
    '''
    strides = [1, 1, 1, 1]  # 卷积时在图像每一维的步长,一维向量[1, strides, strides, 1]，第一位和最后一位固定为 1
    padding = 'SAME' # 考虑边界，不足时用0填充周围；'VALID'不考虑边界
    
    ## 计算两个维度的梯度的网络
    # tf.nn.conv2d返回一个Tensor; tf.abs(tensor)对tensor中每个元素求绝对值
    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding)) # | I^[i,j]-I^[i-1,j] |
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding)) # | I^[i,j]-I^[i,j-1] |
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding)) # | I[i,j]-I[i-1,j] |
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding)) # | I[i,j]-I[i,j-1] |
    # 梯度差
    grad_diff_x = tf.abs(gt_dx - gen_dx) #  x维度的差值
    grad_diff_y = tf.abs(gt_dy - gen_dy) #  y维度的差值

    # condense into one tensor and avg 压缩
    # tf.reduce_mean()计算均值，未指定axis则计算tensor所有维度元素的均值，返回一个值
    return tf.reduce_mean(grad_diff_x ** alpha + grad_diff_y ** alpha) # alpha=1

