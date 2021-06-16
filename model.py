
import tensorflow as tf
import tensorflow.keras.layers as layers


def create_model(input_shape, n_classes, weights=None):
    scaling = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*input_shape),
        layers.experimental.preprocessing.Rescaling(1/255),
    ])

    # Dropout makes images too noisy
    augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        # layers.experimental.preprocessing.RandomZoom(0.5),
        layers.experimental.preprocessing.RandomTranslation(0.3, 0.3),
        layers.experimental.preprocessing.RandomRotation(1 / 12),
    ])

    convolution = tf.keras.Sequential([
        # large patterns
        layers.Conv2D(filters=256, kernel_size=16, strides=(4, 4), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.L1(1e-4)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        # medium patterns
        layers.Conv2D(filters=256, kernel_size=8, strides=(2, 2), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.L1(1e-4)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        # small patterns
        layers.Conv2D(filters=256, kernel_size=4, strides=(1, 1), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.L1(1e-4)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    ])

    mlp = tf.keras.Sequential([
        layers.Dense(units=1024, activation="relu"),
        layers.Dense(units=512, activation="relu"),
        layers.Dense(units=256, activation="relu"),
        layers.Dense(units=128, activation="relu"),
        layers.Dense(units=64, activation="relu"),
    ])

    output = layers.Dense(units=n_classes, activation=None)

    model = tf.keras.Sequential([
        scaling,
        augmentation,
        convolution,
        layers.Flatten(),
        mlp,
        output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    if weights is not None:
        # 3 input channels: R G B
        dummy_instance = tf.zeros((1, *input_shape, 3))
        model(dummy_instance)
        model.load_weights(weights)

    return model
