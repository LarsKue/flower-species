
import tensorflow as tf
import pathlib as pl
import matplotlib.pyplot as plt

from model import create_model
from filters import visualize_filters


def data_set(path, image_size, batch_size):
    train = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=0,
        image_size=image_size,
        batch_size=batch_size,
    )

    train.shuffle(buffer_size=batch_size, seed=0)
    train.cache()

    val = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=0,
        image_size=image_size,
        batch_size=batch_size,
    )

    val.cache()

    return train, val


def main(weights=None):
    path = pl.Path("data")
    image_size = (180, 180)
    batch_size = 64

    train, val = data_set(path, image_size, batch_size)

    n_classes = len(train.class_names)

    model = create_model(image_size, n_classes, weights=weights)

    scaling = model.layers[0]
    augmentation = model.layers[1]

    # base image
    image, _ = next(iter(train))
    image = image[0]
    image = tf.expand_dims(image, 0)

    # first plot
    img = image[0].numpy().astype("uint8")

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(img, origin="lower")
    plt.axis("off")

    for i in range(1, 9):
        # augment image
        aug = augmentation(image)
        aug = aug[0].numpy().astype("uint8")
        plt.subplot(3, 3, i + 1)
        plt.imshow(aug, origin="lower")
        plt.axis("off")

    plt.show()

    history = model.fit(
        train,
        validation_data=val,
        epochs=128,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(verbose=1, min_delta=1e-5, factor=0.1, patience=10),
            # tf.keras.callbacks.EarlyStopping(min_delta=1e-6, patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                "checkpoints/{epoch:02d}.hdf5",
                save_freq="epoch",
                period=8,
                save_weights_only=True,
            )
        ]
    )

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    axes[0].plot(epochs, history.history["loss"], label="Training")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.history["accuracy"], label="Training")
    axes[1].plot(epochs, history.history["val_accuracy"], label="Validation")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
    # visualize_filters()
