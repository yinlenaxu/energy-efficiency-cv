import json
import os

import numpy as np
from codecarbon import EmissionsTracker
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.datasets import cifar10, mnist
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

# config
path = os.getcwd()
with open(os.path.join(path, 'config.json')) as json_file:
    config = json.load(json_file)

if __name__ == "__main__":
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    # Load data
    if config['input'] == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif config["input"] == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Create extra dimension and scale image
        x_train = np.dstack([x_train] * 3)
        x_train = x_train.reshape(-1, 28, 28, 3)
        x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in x_train])
        x_test = np.dstack([x_test] * 3)
        x_test = x_test.reshape(-1, 28, 28, 3)
        x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in x_test])

    # Input image dimensions and classes
    input_shape = x_train.shape[1:]

    # Target one-hot vectorization
    y_train = keras.utils.to_categorical(y_train, config['num_classes'])
    y_test = keras.utils.to_categorical(y_test, config['num_classes'])

    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
    )  # randomly flip images

    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    base_model = None
    model = None

    # Import pre-trained layers
    if config["model"] == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif config["model"] == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif config['model'] == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    # Establish new fully connected block
    x = base_model.output
    x = Flatten()(x)  # flatten from convolution tensor output
    x = Dense(4096, activation='relu')(x)  # number of layers and units are hyperparameters, as usual
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(config['num_classes'], activation=config['activation'])(x)  # should match # of classes predicted

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['accuracy'])

    # CodeCarbon Tracker
    tracker = EmissionsTracker()

    # Training
    for i in range(config['runs']):
        tracker.start()
        history_vgg16 = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=config['batch_size']),
            epochs=config['epochs'],
            verbose=1,
            validation_data=(x_test, y_test)
        )
        tracker.stop()
