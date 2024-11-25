# Import necessary libraries
import matplotlib.pyplot as plt  # Used for generating plots
import tensorflow as tf  # Core library for machine learning
import pandas as pd  # Used for data manipulation and analysis (CSV)
import pathlib  # Used for path manipulation

# Import libraries from TensorFlow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

from pathlib import Path

# Download and prepare the dataset
dataset_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/bycbh73438-1.zip"
data_dir = tf.keras.utils.get_file('Dataset.zip', origin=dataset_url, cache_subdir=r'path to save the dataset')
data_dir = pathlib.Path(data_dir).with_suffix('')

batch_size = 32
img_height = 128
img_width = 128

# Split data into training and validation sets using image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE  # Set a hint for automatic tuning

# Preprocess training data (cache, shuffle, prefetch)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define data augmentation layers for random transformations
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1)
    ]
)


def model_compile(model):
    """Compiles the model with Adam optimizer, SparseCategoricalCrossentropy loss, and accuracy metric."""
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def model_build(num_classes, data_augmentation, dropout_rate, num_conv_layers):
    """Builds a Sequential model based on the provided parameters."""
    model = Sequential([
        # data_augmentation, uncomment to include data augmentation
        layers.Rescaling(1./255),
        # List comprehension for convolutional layers
        *(layer for i in range(num_conv_layers) for layer in [layers.Conv2D(16 * 2**i, 3, padding='same', activation='relu'), layers.MaxPooling2D()])  
    ])
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, name="outputs"))
    return model


def model_training_saving(model, train_ds, val_ds, epochs, description):
    """Trains the model and saves the weights of the best epoch."""

    path_keras = r'root path to save the model ' + description + '.keras'

    # Create ModelCheckpoint callback to save the best weights
    # ... (previous code)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=path_keras,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint_callback]
    )

    return history

num_classes = len(class_names)

num_layers = 3

dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]

epochs = 20

for dropout in dropouts:
    for layer in range(num_layers):
        layer = layer + 3  # Creates models with 3, 4, and 5 convolutional layers
        description = f'Dropout {dropout*100}% - {layer} Layers +'
        print("Execution model " + description)
        model = model_build(num_classes, data_augmentation, dropout, layer)
        model = model_compile(model)
        model.build((None, 128, 128, 3))
        model.summary()
        history = model_training_saving(model, train_ds, val_ds, epochs, description)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        max_val_acc = max(val_acc) * 100
        plt.plot(epochs_range, val_acc, label='Validation Accuracy - Best: ' + f'{max(val_acc) * 100:.2f}%')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy - Model ' + description)

        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        min_val_loss = min(val_loss)
        plt.plot(epochs_range, val_loss, label='Validation Loss - Best: ' + f'{min(val_loss):.2f}')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss - Model ' + description)

        path_fig = r'root path to save the chart ' + description + '.png'
        plt.savefig(path_fig)
        plt.show()

        # Extraction of training results data
        data = {
            'Epoch': epochs_range,
            'Training Accuracy': acc,
            'Validation Accuracy': val_acc,
            'Training Loss': loss,
            'Validation Loss': val_loss
        }

        # Creates a Pandas DataFrame
        df = pd.DataFrame(data)

        path_csv = r'root path to save the csv' + description + '.csv'
        df.to_csv(path_csv, index=False)

"""
Excerpt used to retrain the models

# Path to the folder with the pre-trained models
path_models = Path(r'path to model')

epochs = 20


# Iterating through the files
for path_model in path_models.rglob('*'):
    model = keras.models.load_model(path_model)
    loss, accuracy = model.evaluate(val_ds)
    print('Loss:', loss)
    print('Accuracy:', accuracy)
    _, _, description = path_model.stem.partition(' ')

    print("Retraining Model " + description)

    history = model_training_saving(model, train_ds, val_ds, epochs, description)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    max_val_acc = max(val_acc) * 100
    plt.plot(epochs_range, val_acc, label='Validation Accuracy - Best: ' + f'{max(val_acc) * 100:.2f}%')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy - Model ' + description)

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    min_val_loss = min(val_loss)
    plt.plot(epochs_range, val_loss, label='Validation Loss - Best: ' + f'{min(val_loss):.2f}')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss - Model ' + description)

    path_fig = r'root path to save the chart ' + description + '.png'
    plt.savefig(path_fig)
    plt.show()

    # Extraction of training results data
    data = {
        'Epoch': epochs_range,
        'Training Accuracy': acc,
        'Validation Accuracy': val_acc,
        'Training Loss': loss,
        'Validation Loss': val_loss
    }

    # Creates a Pandas DataFrame
    df = pd.DataFrame(data)

    path_csv = r'root path to save the csv ' + description + '.csv'
    df.to_csv(path_csv, index=False)
"""