import tensorflow as tf

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Generator2():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(512, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 512)
        downsample(512, 4),  # (batch_size, 64, 64, 512)
        downsample(512, 4),  # (batch_size, 32, 32, 512)
        downsample(256, 4),  # (batch_size, 16, 16, 256)
        downsample(128, 4),  # (batch_size, 8, 8, 128)
        downsample(64, 4),  # (batch_size, 4, 4, 64)
        downsample(32, 4),  # (batch_size, 2, 2, 32)
        downsample(32, 4),  # (batch_size, 1, 1, 32)
    ]

    up_stack = [
        upsample(32, 4, apply_dropout=True),  # (batch_size, 2, 2, 2*32)
        upsample(64, 4, apply_dropout=True),  # (batch_size, 4, 4, 2*64)
        upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 8, 2*128)
        upsample(256, 4),  # (batch_size, 16, 16, 2*256)
        upsample(512, 4),  # (batch_size, 32, 32, 2*512)
        upsample(512, 4),  # (batch_size, 64, 64, 2*512)
        upsample(512, 4),  # (batch_size, 128, 128, 2*512)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Generator3():
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    down_stack = [
        downsample(512, 4, apply_batchnorm=False),  # (batch_size, 256, 256, 512)
        downsample(256, 4),  # (batch_size, 128, 128, 256)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(128, 4),  # (batch_size, 32, 32, 128)
        downsample(64, 4),  # (batch_size, 16, 16, 64)
        downsample(64, 4),  # (batch_size, 8, 8, 64)
        downsample(32, 4),  # (batch_size, 4, 4, 32)
        downsample(32, 4),  # (batch_size, 2, 2, 32)
        downsample(16, 4)   # (batch_size, 1, 1, 16)
    ]

    up_stack = [
        upsample(32, 4, apply_dropout=True),  # (batch_size, 2, 2, 2*32)
        upsample(32, 4, apply_dropout=True),  # (batch_size, 4, 4, 2*32)
        upsample(64, 4, apply_dropout=True),  # (batch_size, 8, 8, 2*64)
        upsample(64, 4),  # (batch_size, 16, 16, 2*64)
        upsample(128, 4),  # (batch_size, 32, 32, 2*128)
        upsample(128, 4),  # (batch_size, 64, 64, 2*128)
        upsample(256, 4),  # (batch_size, 128, 128, 2*216)
        upsample(512, 4),  # (batch_size, 256, 256, 2*512)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 512, 512, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)