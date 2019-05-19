import os
import numpy as np
import datetime
import argparse
import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator

from denois_model import get_denois_model, PSNR
from generator import TrainGenerator, ValGenerator


def get_args():
    parser = argparse.ArgumentParser(description="train denoiser model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", type=str, required=False,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=False,
                        help="test image dir")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    # parser.add_argument("--steps", type=int, default=1500,
    parser.add_argument("--steps", type=int, default=1500,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; 'mse' or 'mae' is expected")
    parser.add_argument("--output_path", type=str, default=".",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="-1.5,0.0",
                        help="noise model for source images")
    parser.add_argument("--val_noise_model", type=str, default="-1.5,0.0",
                        help="noise model for validation source images")
    args = parser.parse_args()
    return args


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 5e-4
    if epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    args = get_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_path = args.output_path
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    now = datetime.datetime.now()
    params = str(lr) + "_" + str(now.day) + "." + str(now.month) + "." + str(now.year)
    checkpoint_path = os.path.join(output_path, params)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # def my_categorical_crossentropy(y_true, y_pred):
    #     return categorical_crossentropy(y_true[1], y_pred[1])

    # def my_accuracy(y_true, y_pred):
    #     return categorical_accuracy(y_true[1], y_pred[1])

    model = get_denois_model()
    model.compile(optimizer=Adam(lr=lr_schedule(0)),
                  loss={"UNet_denoiser": "mae", "ResNet20v1_classificator": "categorical_crossentropy"},
                  loss_weights={"UNet_denoiser": 0., "ResNet20v1_classificator": 1.},
                  metrics={"UNet_denoiser": PSNR, "ResNet20v1_classificator": "accuracy"})
    model.summary()

    # source_noise_model = get_noise_model(args.source_noise_model)
    # val_noise_model = get_noise_model(args.val_noise_model)
    train_generator = TrainGenerator(x_train, y_train, batch_size)
    val_generator = ValGenerator(x_test, y_test)

    callbacks = [
        LearningRateScheduler(schedule=lr_schedule),
        ModelCheckpoint(str(checkpoint_path) +
                        "/weights.{epoch:03d}-{val_loss:.3f}-{loss:.3f}.hdf5",
                        monitor="val_ResNet20v1_classificator_acc",
                        verbose=1,
                        mode="min",
                        save_best_only=True),
        ReduceLROnPlateau(factor=np.sqrt(0.1),
                          cooldown=0,
                          patience=5,
                          min_lr=0.5e-6)
    ]

    hist = model.fit_generator(generator=train_generator,
                               validation_data=val_generator,
                               epochs=nb_epochs, verbose=1, workers=4,
                               callbacks=callbacks)

    np.savez(os.path.join(checkpoint_path, "history.npz"), history=hist.history)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == "__main__":
    main()
