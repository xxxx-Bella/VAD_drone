import tensorflow as tf
# from tensorflow.contrib.layers import conv2d, max_pool2d, conv2d_transpose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose


# generator

def unet(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=1):
    """
    :param inputs: input tensor, shape[None, height, width, channel]
    :param layers: number of layers
    :param features_root: number of features in the first layer （第一层提取64个特征）
    :param filter_size: size of each conv layer (kernel_size = 3x3)
    :param pool_size: size of each max pooling layer
    :param output_channel: number of channel for output tensor
    :return: a tensor, shape[None, height, width, output_channel]
    """

    # in_node = inputs
    conv = []  # 用于concatenate

    '''encoder 逐层卷积池化,最后一层不池化'''
    # 第0层
    x = Conv2D(filters=64, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(inputs)
    x = Conv2D(filters=64, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    conv.append(x)  # conv[0]
    # print('ENcoder: layer=0; conv[0].shape=', x.shape)  # (1, 256, 256, 64)
    x = MaxPooling2D(pool_size=pool_size, padding='SAME')(x)
    # print('ENcoder: layer=0; x_shape=', x.shape)  # (1, 128, 128, 64)
    # 第1层
    x = Conv2D(filters=128, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    x = Conv2D(filters=128, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    conv.append(x)  # conv[1]
    # print('ENcoder: layer=0; conv[1].shape=', x.shape)  # (1, 128, 128, 128)
    x = MaxPooling2D(pool_size=pool_size, padding='SAME')(x)
    # print('ENcoder: layer=1; x_shape=', x.shape)  # (1, 64, 64, 128)
    # 第2层
    x = Conv2D(filters=256, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    x = Conv2D(filters=256, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    conv.append(x)  # conv[2]
    # print('ENcoder: layer=0; conv[2].shape=', x.shape)  # (1, 64, 64, 256)
    x = MaxPooling2D(pool_size=pool_size, padding='SAME')(x)
    # print('ENcoder: layer=2; x_shape=', x.shape)  # (1, 32, 32, 256)
    # 第3层
    x = Conv2D(filters=512, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    x = Conv2D(filters=512, kernel_size=filter_size, padding = 'SAME',activation = 'relu')(x)
    conv.append(x)  # conv[3]
    # print('ENcoder: layer=0; conv[3].shape=', x.shape)  # (1, 32, 32, 512)

    # ####################### original
    # for layer in range(0, layers): # 若layers=4,则0,1,2,3层
    #     features = 2**layer*features_root # 每层输出的filter数量（提取的特征个数）
    #     # 2**0*64=64 -> 2**1*64=128 -> 2**2*64=256 -> 2**3*64=512
    #     conv1 = Conv2D(inputs=in_node, num_outputs=features, kernel_size=filter_size)  # stride=1        
    #     conv2 = Conv2D(inputs=conv1, num_outputs=features, kernel_size=filter_size)
    #     conv.append(conv2)

    #     if layer < layers - 1: # 若layers=4, 则0,1,2层（最后一层不池化）
    #         in_node = MaxPooling2D(inputs=conv2, kernel_size=pool_size, padding='SAME')  # 作为下一层的输入
    # in_node = conv[-1]  # encoder的最后一层的conv2作为decoder的输入
    #########################

    '''decoder 逐层：反卷积并连接，再对连接后的特征图进行两次卷积操作'''
    # 第2层
    deconv = Conv2DTranspose(filters=256, kernel_size=pool_size, strides=pool_size)(x)  # 512//2
    # print('DEcoder: layer=2; deconv_shape=', deconv.shape)  # (1, 64, 64, 256)
    deconv_concat = tf.concat([conv[2], deconv], axis=3)
    # print('DEcoder: layer=2; deconv_concat_shape=', deconv_concat.shape)  # (1, 64, 64, 512)

    x = Conv2D(filters=256, kernel_size=filter_size, padding = 'SAME')(deconv_concat)
    # print('DEcoder: layer=2; x_shape=', x.shape)   # (1, 64, 64, 256)
    x = Conv2D(filters=256, kernel_size=filter_size, padding = 'SAME')(x)
    # print('DEcoder: layer=2; x_shape=', x.shape)  # (1, 60, 60, 256)
    # 第1层
    deconv = Conv2DTranspose(filters=128, kernel_size=pool_size, strides=pool_size)(x)  # 256//2
    # print('DEcoder: layer=1; deconv_shape=', deconv.shape)  # (1, 120, 120, 128)
    deconv_concat = tf.concat([conv[1], deconv], axis=3)  # input:[1,128,128,128], [1,120,120,128]  ERROR !! 
    # print('DEcoder: layer=2; deconv_concat_shape=', deconv_concat.shape)  # 
    x = Conv2D(filters=128, kernel_size=filter_size, padding = 'SAME')(deconv_concat)
    x = Conv2D(filters=128, kernel_size=filter_size, padding = 'SAME')(x)
    # print('DEcoder: layer=1; x_shape=', x.shape)
    # 第0层
    deconv = Conv2DTranspose(filters=64, kernel_size=pool_size, strides=pool_size)(x) # 128//2
    deconv_concat = tf.concat([conv[0], deconv], axis=3)
    x = Conv2D(filters=64, kernel_size=filter_size, padding = 'SAME')(deconv_concat)
    x = Conv2D(filters=64, kernel_size=filter_size, padding = 'SAME')(x)
    # print('DEcoder: layer=0; x_shape=', x.shape)
    # ######################## original
    # for layer in range(layers-2, -1, -1): # 若layers=4,则2,1,0层
    #     features = 2**(layer+1)*features_root
    #     # 2^3*64=512 -> 2^2*64=256 -> 2^1*64=128
    #     h_deconv = Conv2DTranspose(inputs=in_node, num_outputs=features//2, kernel_size=pool_size, stride=pool_size)
    #     h_deconv_concat = tf.concat([conv[layer], h_deconv], axis=3)

    #     conv1 = Conv2D(inputs=h_deconv_concat, num_outputs=features//2, kernel_size=filter_size)
    #     in_node = Conv2D(inputs=conv1, num_outputs=features//2, kernel_size=filter_size)  # 作为下一层的输入
    #########################

    '''最后再卷积一次 输出的channel=3 (RGB)'''
    output = Conv2D(filters=output_channel, kernel_size=filter_size, padding = 'SAME', activation=None)(x)
    output = tf.tanh(output)

    # ######################## original
    # output = Conv2D(inputs=in_node, num_outputs=output_channel, kernel_size=filter_size, activation_fn=None)  # output_channel=3(models.py); activation_fn=None: 跳过它并保持线性激活)
    # output = tf.tanh(output)  # 最后进行激活，将范围映射到(-1,1)
    #########################

    return output  # shape=[None, height, width, output_channel]=[None,256,256,3]




