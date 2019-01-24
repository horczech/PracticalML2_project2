import numpy as np
import keras
import cv2
import math

BBOX_PARAMS_COUNT = 5

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_list, batch_size=32, shuffle=True, X_shape=(448, 448, 3), y_shape=735, grid_size=7, class_count=10):
        """Initialization"""

        self.data_infos = data_list
        self.batch_size = batch_size
        self.is_shuffle = shuffle
        self.y_shape = y_shape
        self.X_shape = X_shape
        self.grid_size = grid_size
        self.class_count = class_count
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return int(np.ceil(len(self.data_infos) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.indexes = np.arange(len(self.data_infos))
        if self.is_shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""

        # Initialization
        X = np.empty((self.batch_size, self.X_shape[0], self.X_shape[1], self.X_shape[2]))
        y = np.empty((self.batch_size, self.grid_size, self.grid_size, BBOX_PARAMS_COUNT+self.class_count), dtype=float)

        # Generate data
        for i, idx in enumerate(indexes):
            data_info = self.data_infos[idx]

            # load image
            img_path = str(data_info['image_path'])
            img = cv2.imread(img_path)
            # ToDo: This is not probably a best place to resize the image and image
            # img = cv2.resize(img, (self.X_shape[0], self.X_shape[1]))
            # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # load labels (result)
            target = self.parse_object_data(data_info['objects'])

            # store sample
            X[i,] = img

            # Store class
            y[i] = target

        return X, y


    def parse_object_data(self, object_infos):
        target_tensor = np.zeros((self.grid_size, self.grid_size, BBOX_PARAMS_COUNT + self.class_count))

        for bbox in object_infos:
            # ToDo: what to do when multiple objects will be multiple objects in one cell
            width = bbox['width']
            height = bbox['height']

            box_confidence_score = 1

            class_id = bbox['class_id']
            class_probabilities = np.zeros(shape=self.class_count)
            class_probabilities[class_id] = 1

            x, x_cell_idx = math.modf(bbox['x_center'] * self.grid_size)

            y, y_cell_idx = math.modf(bbox['y_center'] * self.grid_size)

            # (x, y, w, h, box confidence score, class probabilities)
            object_tensor = np.array([x, y, width, height, box_confidence_score])
            object_tensor = np.append(object_tensor, class_probabilities)
            target_tensor[int(x_cell_idx), int(y_cell_idx), :] = object_tensor

        return target_tensor