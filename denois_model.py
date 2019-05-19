import tensorflow as tf
# from keras.backend import set_session
import keras.backend as K
from keras.models import Model
from keras.layers import Input, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout, Subtract
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from classification_model import resnet_v1


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    # y_pred = K.clip(y_pred, 0.0, max_pixel)
    return 10.0 * tf_log10((max_pixel ** 2) / K.mean(K.square(y_pred - y_true)))


def get_denois_model():
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # set_session(tf.Session(config=config))
    return get_unet_model()


# UNet:
def get_unet_model(input_channel_num=3, out_ch=3, start_ch=64, depth=4, inc_rate=2.,
                   dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    def _conv_block(m, dim, bn, res, do=0):
        n = Conv2D(dim, 3, padding='same', kernel_initializer="he_normal")(m)
        n = PReLU(shared_axes=[1, 2])(n)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, padding='same', kernel_initializer="he_normal")(n)
        n = PReLU(shared_axes=[1, 2])(n)
        n = BatchNormalization()(n) if bn else n

        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, padding='same', kernel_initializer="he_normal")(m)
                m = PReLU(shared_axes=[1, 2])(n)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, padding='same', kernet_initsalzer="he_normal")(m)
                m = PReLU(shared_axes=[1, 2])(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, bn, res)
        else:
            m = _conv_block(m, dim, bn, res, do)

        return m

    i = Input(shape=(32, 32, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, dropout, batchnorm, maxpool, upconv, residual)
    denois_res = Conv2D(out_ch, 1, kernel_initializer="he_normal", name="UNet_denoiser")(o)

    classif_model = resnet_v1()
    for layer in classif_model.layers:
        layer.trainable = False
    classif_res = classif_model(denois_res)

    model = Model(inputs=i, outputs=[denois_res, classif_res])
    return model
