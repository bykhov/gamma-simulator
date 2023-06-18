from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, \
    Conv2D, MaxPooling2D, BatchNormalization, Input, Concatenate, \
    GlobalAveragePooling2D, GlobalMaxPool2D, Conv1D, \
    LSTM, Bidirectional, Reshape, Softmax, MaxPooling1D

import tensorflow as tf

# %% Model
def Conv_Block(inputs, step, n_filters, kernel_size=3, dropout_prob=0.0):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(n_filters * 2 ** step, kernel_size, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(n_filters * 2 ** step, kernel_size, padding='causal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    conv = tf.keras.layers.Activation('relu')(x)

    conv = Dropout(dropout_prob)(conv)

    x = MaxPooling1D(2, strides=2)(conv)
    return x, conv


def trans_conv1D(inputs, contractive_inputs, step, n_filters, kernel_size=3):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(n_filters * 2 ** step,
                                        kernel_size, strides=2, padding='same')(inputs)  # Stride = 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = Concatenate(axis=-1)([x, contractive_inputs])

    x = tf.keras.layers.Conv1D(n_filters * 2 ** step, kernel_size, padding='causal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(n_filters * 2 ** step, kernel_size, padding='causal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def unet_model4(frame_length, n_filters=16):
    inputs = Input(shape=(frame_length,))
    inputs = tf.expand_dims(inputs, axis=-1)
    cblock0 = Conv_Block(inputs=inputs, step=0, n_filters=n_filters)  # n_filters = 32
    cblock1 = Conv_Block(inputs=cblock0[0], step=1, n_filters=n_filters, dropout_prob=0.15)  # n_filters = 32*2
    cblock2 = Conv_Block(inputs=cblock1[0], step=2, n_filters=n_filters, dropout_prob=0.15)  # n_filters = 32*4
    cblock3 = Conv_Block(inputs=cblock2[0], step=3, n_filters=n_filters, dropout_prob=0.15)  # n_filters = 32*8
    cblock4 = Conv_Block(inputs=cblock3[0], step=4, n_filters=n_filters, dropout_prob=0.15)  # n_filters = 32*8

    ublock6 = trans_conv1D(cblock4[1], cblock3[1], step=3, n_filters=n_filters)
    ublock7 = trans_conv1D(ublock6, cblock2[1], step=2, n_filters=n_filters)
    ublock8 = trans_conv1D(ublock7, cblock1[1], step=1, n_filters=n_filters)
    ublock9 = trans_conv1D(ublock8, cblock0[1], step=0, n_filters=n_filters)

    conv10 = Conv1D(n_filters, kernel_size=3, padding='causal')(ublock9)
    utputs = Dense(6, activation='softmax')(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def unet_model3(frame_length, n_filters=16):
    inputs = Input(shape=(frame_length,))
    inputs = tf.expand_dims(inputs, axis=-1)
    cblock0 = Conv_Block(inputs=inputs, step=0, n_filters=n_filters)  # n_filters = 32
    cblock1 = Conv_Block(inputs=cblock0[0], step=1, n_filters=n_filters, dropout_prob=0.15)  # n_filters = 32*2
    cblock2 = Conv_Block(inputs=cblock1[0], step=2, n_filters=n_filters, dropout_prob=0.15)  # n_filters = 32*4
    cblock3 = Conv_Block(inputs=cblock2[0], step=3, n_filters=n_filters, dropout_prob=0.2)  # n_filters = 32*8

    ublock7 = trans_conv1D(cblock3[1], cblock2[1], step=2, n_filters=n_filters)
    ublock8 = trans_conv1D(ublock7, cblock1[1], step=1, n_filters=n_filters)
    ublock9 = trans_conv1D(ublock8, cblock0[1], step=0, n_filters=n_filters)

    conv10 = Conv1D(n_filters, kernel_size=3, padding='causal')(ublock9)
    outputs = Dense(6, activation='softmax')(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print('U-Net 3 Model')
    return model


def unet_model2(frame_length, n_filters=16):
    inputs = Input(shape=(frame_length,))
    inputs = tf.expand_dims(inputs, axis=-1)
    cblock0 = Conv_Block(inputs=inputs, step=0, n_filters=n_filters)  # n_filters = 32
    cblock1 = Conv_Block(inputs=cblock0[0], step=1, n_filters=n_filters, dropout_prob=0.1)  # n_filters = 32*2
    cblock2 = Conv_Block(inputs=cblock1[0], step=2, n_filters=n_filters, dropout_prob=0.1)  # n_filters = 32*4
    # cblock3 = Conv_Block(inputs=cblock2[0], step=3, n_filters=n_filters, dropout_prob=0.3)  # n_filters = 32*8

    # ublock7 = trans_conv1D(cblock3[1], cblock2[1], step=2, n_filters=n_filters)
    ublock8 = trans_conv1D(cblock2[1], cblock1[1], step=1, n_filters=n_filters)
    ublock9 = trans_conv1D(ublock8, cblock0[1], step=0, n_filters=n_filters)

    conv10 = Conv1D(n_filters, kernel_size=3, padding='causal')(ublock9)
    outputs = Dense(6, activation='softmax')(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
