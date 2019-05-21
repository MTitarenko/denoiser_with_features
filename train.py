import os
import numpy as np
import datetime
import argparse
import keras
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10

from denois_model import get_denois_model, PSNR
from generator import TrainGenerator, ValGenerator
from noise_model import get_noise_model


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
    parser.add_argument("--lr", type=float, default=0.0000005,
                        help="learning rate")
    parser.add_argument("--output_path", type=str, default=".",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="-2.0,-0.5",
                        help="noise model for source images")
    parser.add_argument("--val_noise_model", type=str, default="-2.0,-0.5",
                        help="noise model for validation source images")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        lr = self.initial_lr
        if epoch_idx > self.epochs * 0.9:
            lr *= 0.03125
        elif epoch_idx > self.epochs * 0.8:
            lr *= 0.0625
        elif epoch_idx > self.epochs * 0.6:
            lr *= 0.125
        elif epoch_idx > self.epochs * 0.4:
            lr *= 0.25
        elif epoch_idx > self.epochs * 0.2:
            lr *= 0.5
        print('Learning rate: ', lr)
        return lr


def main():
    args = get_args()
    output_path = args.output_path
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test -= x_train_mean
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    now = datetime.datetime.now()
    params = "denoiser_" + str(lr) + "_" + str(now.day) + "." + str(now.month) + "." + str(now.year)
    checkpoint_path = os.path.join(output_path, params)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    model = get_denois_model()
    model.compile(optimizer=Adam(lr=lr),
                  loss={"UNet_denoiser": "mse", "ResNet20v1_classificator": "categorical_crossentropy"},
                  loss_weights={"UNet_denoiser": 0., "ResNet20v1_classificator": 1.},
                  metrics={"UNet_denoiser": PSNR, "ResNet20v1_classificator": "accuracy"})
    model.summary()

    train_noise_model = get_noise_model(args.source_noise_model)
    val_noise_model = get_noise_model(args.val_noise_model)
    train_generator = TrainGenerator(x_train, y_train, batch_size, train_noise_model, x_train_mean)
    val_generator = ValGenerator(x_test, y_test, val_noise_model, x_train_mean)

    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
        ModelCheckpoint(str(checkpoint_path) +
                        "/weights.{epoch:03d}-{ResNet20v1_classificator_acc:.3f}-{val_ResNet20v1_classificator_acc:.3f}-{val_UNet_denoiser_PSNR:.1f}.hdf5",
                        monitor="val_ResNet20v1_classificator_acc",
                        verbose=1,
                        mode="max",
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


if __name__ == "__main__":
    main()
