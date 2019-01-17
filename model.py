import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, \
    MaxPool2D, ReLU, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from utils import calculate_IOU


import cv2
import numpy as np
import math

BBOX_PARAMS_COUNT = 5




class YOLO:
    def __init__(self,
                 input_size,
                 grid_size,
                 bbox_count,
                 classes,
                 lambda_coord=5,
                 lambda_noobj=0.5
                 ):
        self.input_size = input_size
        self.grid_size = grid_size
        self.bbox_count = bbox_count
        self.classes = classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


        # grid_cell_count_row x grid_cell_count_coll x bboxes_count_per_cell + class_count
        output_layer_size = self.grid_size * self.grid_size * (self.bbox_count * BBOX_PARAMS_COUNT + len(self.classes))

        self.model = self.build_yolo_model(output_layer_size)

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


        # ToDo: I coppied that line -> should research that
        # optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # self.model.compile(loss=self.custom_loss, optimizer='adam')
        self.model.compile(loss=self.custom_loss, optimizer='adam')

        self.model.fit(x_train, y_train, epochs=1, batch_size=1)


    def custom_loss(self, y_true, y_pred):

        y_true_shape = (self.grid_size, self.grid_size, BBOX_PARAMS_COUNT+len(self.classes))
        y_pred_shape = (self.grid_size, self.grid_size, self.bbox_count*BBOX_PARAMS_COUNT+len(self.classes))

        y_true = tf.reshape(y_true, y_true_shape)
        y_pred = tf.reshape(y_pred, y_pred_shape)

        y_true = tf.Print(y_true, [y_true[3, 3, :], y_pred[3, 3, :]], message='\n\ny_true, y_pred: ', summarize=100)

        # # # # # # # # # # # # # # # # # # #
        # parse data
        # # # # # # # # # # # # # # # # # # #

        # 0 , 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, ..., pn
        # x1, y1, w1, h1, C1, x2, y2, w2, h2, C2, p1, p2, ..., pn
        predicted_bbox_1 = y_pred[:, :, :5]
        predicted_bbox_2 = y_pred[:, :, 5:10]
        predicted_class_prob = y_pred[:, :, 10:]

        true_box = y_true[:, :, :4]
        true_object_confidence = y_true[:, :, 4]
        true_class_prob = y_true[:, :, 5:]

        # # # # # # # # # # # # # # # # # # #
        # find responsible bboxes
        # # # # # # # # # # # # # # # # # # #

        iou_bbox1 = calculate_IOU(predicted_bbox_1, true_box)
        iou_bbox2 = calculate_IOU(predicted_bbox_2, true_box)

        responsible_pred_bbox = tf.greater(iou_bbox1, iou_bbox2)

        responsible_pred_bbox = tf.tile(tf.expand_dims(responsible_pred_bbox, axis=2), [1, 1, 5])

        responsible_pred_bbox = tf.where(responsible_pred_bbox, predicted_bbox_1, predicted_bbox_2)

        # # # # # # # # # # # # # # # # # # #
        # x,y loss
        # # # # # # # # # # # # # # # # # # #

        # (x - x')^2 + (y - y')^2
        x_loss = tf.squared_difference(true_box[:, :, 0], responsible_pred_bbox[:, :, 0])
        y_loss = tf.squared_difference(true_box[:, :, 1], responsible_pred_bbox[:, :, 1])
        xy_loss = x_loss + y_loss

        # if the object is not present in the cell that the sum is zero
        xy_loss = xy_loss * true_object_confidence
        xy_loss = self.lambda_coord * tf.reduce_sum(xy_loss)

        # # # # # # # # # # # # # # # # # # #
        # w,h loss
        # # # # # # # # # # # # # # # # # # #

        # (sqrt(w) - sqrt(w'))^2 + (sqrt(h) - sqrt(h'))^2
        w_loss = tf.squared_difference(tf.sqrt(true_box[:, :, 2]), tf.sqrt(responsible_pred_bbox[:, :, 2]))
        h_loss = tf.squared_difference(tf.sqrt(true_box[:, :, 3]), tf.sqrt(responsible_pred_bbox[:, :, 3]))
        wh_loss = w_loss + h_loss

        # if the object is not present in the cell that the sum is zero
        wh_loss = wh_loss * true_object_confidence
        wh_loss = self.lambda_coord * tf.reduce_sum(wh_loss)

        # # # # # # # # # # # # # # # # # # #
        # bbox confidence loss
        # # # # # # # # # # # # # # # # # # #

        object_loss = tf.squared_difference(true_object_confidence, responsible_pred_bbox[:, :, 4])
        object_loss = tf.reduce_sum(object_loss * true_object_confidence)

        no_object_loss = tf.squared_difference(true_object_confidence, responsible_pred_bbox[:, :, 4])
        no_object_loss = tf.reduce_sum(self.lambda_noobj * tf.multiply(no_object_loss, 1 - true_object_confidence))

        confidence_loss = object_loss + no_object_loss

        # # # # # # # # # # # # # # # # # # #
        # classification loss
        # # # # # # # # # # # # # # # # # # #
        classification_loss = tf.squared_difference(true_class_prob, predicted_class_prob)
        classification_loss = tf.reduce_sum(classification_loss, axis=2)
        classification_loss = tf.reduce_sum(classification_loss * true_object_confidence)

        # # # # # # # # # # # # # # # # # # #
        # Total loss
        # # # # # # # # # # # # # # # # # # #
        loss = xy_loss + wh_loss + confidence_loss + classification_loss

        # # # # # # # # # # # # # # # # # # #
        # Debug Info
        # # # # # # # # # # # # # # # # # # #
        loss = tf.Print(loss, [xy_loss], message='Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [wh_loss], message='Loss WH \t', summarize=1000)
        loss = tf.Print(loss, [confidence_loss], message='Loss Conf \t', summarize=1000)
        loss = tf.Print(loss, [classification_loss], message='Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)

        return loss

    def build_yolo_model(self, output_layer_size):
        input_image = Input(shape=self.input_size)

        inception_model = InceptionV3(include_top=False,
                                      weights='imagenet',
                                      input_tensor=None,
                                      input_shape=self.input_size,
                                      pooling=None)

        x = inception_model(input_image)


        # Layer 1
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', name='yolo_conv_1')(x)
        x = LeakyReLU(alpha=0.1, name='yolo_relu_1')(x)

        # Layer 2
        x = Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same', name='yolo_conv_2')(x)
        x = LeakyReLU(alpha=0.1, name='yolo_relu_2')(x)

        # Layer 3
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', name='yolo_conv_3')(x)
        x = LeakyReLU(alpha=0.1, name='yolo_relu_3')(x)

        # Layer 4
        x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', name='yolo_conv_4')(x)
        x = LeakyReLU(alpha=0.1, name='yolo_relu_4')(x)

        x = Flatten()(x)


        # Layer 29
        x = Dense(units=4096, name='yolo_dense_1')(x)
        x = LeakyReLU(alpha=0.1, name='yolo_relu_5')(x)

        # Layer 30
        x = Dense(units=output_layer_size, name='yolo_dense_2')(x)
        x = LeakyReLU(alpha=0.0, name='yolo_relu_6')(x)


        model = Model(inputs=input_image, outputs=x)

        return model



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
