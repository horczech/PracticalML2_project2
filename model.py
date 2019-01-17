import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, MaxPool2D, ReLU
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K



import cv2
import numpy as np
import math

BBOX_PARAMS_COUNT = 5




class YOLO:
    def __init__(self,
                 input_size,
                 grid_size,
                 bbox_count,
                 classes
                 ):
        self.input_size = input_size
        self.grid_size = grid_size
        self.bbox_count = bbox_count
        self.classes = classes


        # grid_cell_count_row x grid_cell_count_coll x bboxes_count_per_cell + class_count
        output_layer_size = self.grid_size * self.grid_size * (self.bbox_count * BBOX_PARAMS_COUNT + len(self.classes))
        self.model = self.build_network(self.input_size, output_layer_size)

    def parse_object_data(self, object_infos):
        target_tensor = np.zeros((self.grid_size, self.grid_size, BBOX_PARAMS_COUNT + len(self.classes)))

        for bbox in object_infos:
            # ToDo: what to do when multiple objects will be multiple objects in one cell

            width = bbox['width']
            height = bbox['height']

            box_confidence_score = 1


            class_id = bbox['class_id']
            class_probabilities = np.zeros(shape=len(self.classes))
            class_probabilities[class_id] = 1

            x, x_cell_idx = math.modf(bbox['x_center'] * self.grid_size)

            y, y_cell_idx = math.modf(bbox['y_center'] * self.grid_size)

            # (x, y, w, h, box confidence score, class probabilities)
            object_tensor = np.array([x, y, width, height, box_confidence_score])
            object_tensor = np.append(object_tensor, class_probabilities)
            target_tensor[int(x_cell_idx), int(y_cell_idx), :] = object_tensor

        return target_tensor

    def load_training_data(self, data_infos):
        images = []
        targets = []

        for data_info in data_infos:
            img_path = str(data_info['image_path'])

            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            target = self.parse_object_data(data_info['objects'])

            targets.append(target)
            images.append(img)

        return np.asarray(images), np.asarray(targets)

    def train(self, train_data):
        # ToDo: Make train and validation generators

        x_train, y_train = self.load_training_data(data_infos=train_data)

        y1 = np.random.rand(1, 980)
        y2 = np.random.rand(1, 980)
        y3 = np.random.rand(1, 980)

        y = np.array([y1, y2, y3])

        # ToDo: I coppied that line -> should research that
        # optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # self.model.compile(loss=self.custom_loss, optimizer='adam')
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(x_train, y, epochs=1, batch_size=1)

    def custom_loss(self, y_true, y_pred):
        return K.sum(K.log(y_true) - K.log(y_pred))

    def build_network(self, input_size, output_layer_size):
        input_image = Input(shape=input_size, name='input_layer')

        # Layer 1
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(input_image)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_1')(x)

        # Layer 2
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

        # Layer 3
        x = Conv2D(filters=192, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_2')(x)

        # Layer 4
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

        # Layer 5
        x = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_3')(x)

        # Layer 6
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_4')(x)

        # Layer 7
        x = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_5')(x)

        # Layer 8
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_6')(x)

        # Layer 9
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

        # Layer 10
        x = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_7')(x)

        # Layer 11
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_8')(x)

        # Layer 12
        x = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_9')(x)

        # Layer 13
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_10')(x)

        # Layer 14
        x = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_11')(x)

        # Layer 15
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_12')(x)

        # Layer 16
        x = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_13')(x)

        # Layer 17
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_14')(x)

        # Layer 18
        x = Conv2D(filters=512, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_15')(x)

        # Layer 19
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_16')(x)

        # Layer 20
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

        # Layer 21
        x = Conv2D(filters=512, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_17')(x)

        # Layer 22
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_18')(x)


        # Layer 23
        x = Conv2D(filters=512, kernel_size=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_19')(x)

        # Layer 24
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_20')(x)

        # Layer 25
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_21')(x)

        # Layer 26
        x = Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_22')(x)

        # Layer 27
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_23')(x)

        # Layer 28
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_24')(x)

        # Layer 29
        x = Dense(units=4096)(x)
        x = LeakyReLU(alpha=0.1, name='LeakyRelu_25')(x)

        # Layer 30
        x = Dense(units=output_layer_size)(x)
        x = LeakyReLU(alpha=0.0, name='LeakyRelu_69')(x)

        # x = ReLU(name='ReLU_69')(x)

        return Model(input_image, x)


    def build_network_sequential(self, input_size, output_layer_size):
        model = tf.keras.models.Sequential()

        # Layer 1
        model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=2, input_shape=input_size, data_format="channels_last", border_mode='same', name='conv1'))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 2
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Layer 3
        model.add(Conv2D(filters=192, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 4
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Layer 5
        model.add(Conv2D(filters=128, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 6
        model.add(Conv2D(filters=256, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 7
        model.add(Conv2D(filters=256, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 8
        model.add(Conv2D(filters=512, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 9
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Layer 10
        model.add(Conv2D(filters=256, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 11
        model.add(Conv2D(filters=512, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 12
        model.add(Conv2D(filters=256, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 13
        model.add(Conv2D(filters=512, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 14
        model.add(Conv2D(filters=256, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 15
        model.add(Conv2D(filters=512, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 16
        model.add(Conv2D(filters=256, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 17
        model.add(Conv2D(filters=512, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 18
        model.add(Conv2D(filters=512, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 19
        model.add(Conv2D(filters=1024, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 20
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        # Layer 21
        model.add(Conv2D(filters=512, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 22
        model.add(Conv2D(filters=1024, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))


        # Layer 23
        model.add(Conv2D(filters=512, kernel_size=(1, 1)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 24
        model.add(Conv2D(filters=1024, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 25
        model.add(Conv2D(filters=1024, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 26
        model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=2))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 27
        model.add(Conv2D(filters=1024, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 28
        model.add(Conv2D(filters=1024, kernel_size=(3, 3)))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 29
        model.add(Dense(units=4096))
        model.add(Activation(LeakyReLU(alpha=0.1)))

        # Layer 30
        model.add(Dense(units=output_layer_size))
        model.add(Activation(ReLU))

        return model
