from pathlib import Path
import numpy as np
import cv2
from keras.utils import Sequence


class TrainGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size, noise_model=None, sub=0):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.noise_model = noise_model
        self.sub = sub
        self.len = x_data.shape[0] // batch_size
        self.data_index = np.arange(x_data.shape[0])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx == 0:
            np.random.shuffle(self.data_index)

        y_0 = self.x_data[self.data_index[idx * self.batch_size: (idx + 1) * self.batch_size]]
        y_1 = self.y_data[self.data_index[idx * self.batch_size: (idx + 1) * self.batch_size]]
        if self.noise_model:
            x = np.zeros_like(y_0)
            for i in range(self.batch_size):
                x[i] = self.noise_model(y_0[i]) - self.sub
        else:
            x = y_0 - self.sub
        y_0 -= self.sub
        return x, [y_0, y_1]


class ValGenerator(Sequence):
    def __init__(self, x_data, y_data, noise_model=None, sub=0):
        self.x_data = x_data
        self.y_data = y_data
        self.noise_model = noise_model
        self.sub = sub
        self.len = x_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y_0 = np.expand_dims(x, axis=0) - self.sub
        y_1 = np.expand_dims(self.y_data[idx], axis=0)
        if self.noise_model:
            x = self.noise_model(x) - self.sub
        x = np.expand_dims(x, axis=0)
        return x, [y_0, y_1]


# if __name__ == "__main__":
#     from noise_model import get_noise_model
#     gen = ValGenerator('C:\\coco\\val2014', get_noise_model("-0.3,-0.3"), image_num=3, image_size=256)
#     for i in range(len(gen)):
#         x, y = gen[i]
#         cv2.imshow(str(i) + '_noise', x[0])
#         cv2.imshow(str(i) + '_clean', y[0])
#     cv2.waitKey()
