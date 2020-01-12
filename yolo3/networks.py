from functools import wraps
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add
from keras.regularizers import l2


@wraps(Conv2D)
def conv2d(filters, kernel_size, strides=(1, 1), padding='same', use_bias=False):
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                  kernel_regularizer=l2(5e-4), use_bias=use_bias)


def conv2d_bn_leaky(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = conv2d(filters, kernel_size, strides, padding)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def res_block(x, filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x, filters, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    for _ in range(num_blocks):
        y = conv2d_bn_leaky(x, filters // 2, kernel_size=(1, 1))
        y = conv2d_bn_leaky(y, filters, kernel_size=(3, 3))
        x = Add()([x, y])
    return x


def darknet_body(x):
    x = conv2d_bn_leaky(x, filters=32, kernel_size=(3, 3))
    x = res_block(x, filters=64, num_blocks=1)
    x = res_block(x, filters=128, num_blocks=2)
    x = res_block(x, filters=256, num_blocks=8)
    x = res_block(x, filters=512, num_blocks=8)
    x = res_block(x, filters=1024, num_blocks=4)
    return x


def last_layers(x, filters, out_filters):
    x = conv2d_bn_leaky(x, filters, kernel_size=(1, 1))
    x = conv2d_bn_leaky(x, filters * 2, kernel_size=(3, 3))
    x = conv2d_bn_leaky(x, filters, kernel_size=(1, 1))
    x = conv2d_bn_leaky(x, filters * 2, kernel_size=(3, 3))
    x = conv2d_bn_leaky(x, filters, kernel_size=(1, 1))

    y = conv2d_bn_leaky(x, filters * 2, kernel_size=(3, 3))
    y = conv2d(out_filters, kernel_size=(1, 1), use_bias=True)(y)
    return x, y
