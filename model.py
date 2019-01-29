import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Input, Flatten, Dense, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras import optimizers
from keras.metrics import categorical_accuracy
from utils import calculate_IOU
from data_generator import DataGenerator
import os
import  numpy as np
import cv2
from constants import LOG_FILE_PATH


class YOLO:
    def __init__(self,
                 input_size,
                 grid_size,
                 bbox_count,
                 classes,
                 lambda_coord=5,
                 lambda_noobj=0.5,
                 bbox_params=5):
        self.input_size = input_size
        self.grid_size = grid_size
        self.bbox_count = bbox_count
        self.classes = classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bbox_params = bbox_params

        # grid_cell_count_row x grid_cell_count_coll x bboxes_count_per_cell + class_count
        self.output_layer_size = self.grid_size * self.grid_size * (self.bbox_count * self.bbox_params + len(self.classes))


    def train_gen(self, training_infos, validation_infos, save_model_path, batch_size, nb_epochs, learning_rate, use_pretrained_model, model_name, steps_per_epoch):

        ################################
        # Prepare the model
        ################################
        # if use_pretrained_model is True than rewrite the default model with the model in the save_model_path

        save_model_path = save_model_path + model_name + '.hdf5'

        if use_pretrained_model:
            print(">>>>>> Loading saved model")
            if os.path.isfile(save_model_path):
                self.model = load_model(save_model_path, custom_objects={'custom_loss': self.custom_loss})
            else:
                raise ValueError("No pretrained model found in path: {}".format(save_model_path))
        else:
            # optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # optimizer = optimizers.SGD(lr=1e-16, decay=1e-6, nesterov=True)
            print(">>>>>> Creating new model")
            self.model = self.build_yolo_model(self.output_layer_size)
            optimizer = optimizers.Adam(lr=learning_rate)
            self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ################################
        # Create data generators
        ################################
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'X_shape': self.input_size,
                  'y_shape': self.grid_size * self.grid_size * (self.bbox_params+len(self.classes)),
                  'grid_size': self.grid_size,
                  'class_count': len(self.classes)
                  }

        training_generator = DataGenerator(data_list=training_infos, **params)
        valid_generator = DataGenerator(data_list=validation_infos, **params)

        ################################
        # Create callbacks
        ################################
        checkpoint = ModelCheckpoint(filepath           = save_model_path,
                                     monitor            = 'val_loss',
                                     verbose            = 1,
                                     save_best_only     = True,
                                     save_weights_only  = False,
                                     mode               = 'min',
                                     period             = 1)

        tensorboard = TensorBoard(log_dir=LOG_FILE_PATH+model_name)

        callbacks_list = [checkpoint, tensorboard]

        ################################
        # Train
        ################################
        self.model.fit_generator(generator          = training_generator,
                                 validation_data    = valid_generator,
                                 epochs             = nb_epochs,
                                 callbacks          = callbacks_list,
                                 steps_per_epoch    = steps_per_epoch)

    def custom_acc(self, y_true, y_pred):
        y_true_shape = (-1, self.grid_size, self.grid_size, self.bbox_params + len(self.classes))
        y_pred_shape = (-1, self.grid_size, self.grid_size, self.bbox_count * self.bbox_params + len(self.classes))

        y_true = tf.reshape(y_true, y_true_shape, name='reshape_y_true')
        y_pred = tf.reshape(y_pred, y_pred_shape, name='reshape_y_pred')

        # shape=(?, 21, 21, 10),
        predicted_class_prob = y_pred[:, :, :, 10:]
        true_class_prob = y_true[:, :, :, 5:]

        return categorical_accuracy(true_class_prob, predicted_class_prob)

    def custom_loss(self, y_true, y_pred):

        y_true_shape = (-1, self.grid_size, self.grid_size, self.bbox_params + len(self.classes))
        y_pred_shape = (-1, self.grid_size, self.grid_size, self.bbox_count * self.bbox_params + len(self.classes))
        
        y_true = tf.reshape(y_true, y_true_shape, name='reshape_y_true')
        y_pred = tf.reshape(y_pred, y_pred_shape, name='reshape_y_pred')
        #y_print = tf.Print(y_true, [y_true[0, 10, 10, :], y_pred[0, :, :, :]], message='\n\ny_true, y_pred: ', summarize=10000)

        # # # # # # # # # # # # # # # # # # #
        # parse data
        # # # # # # # # # # # # # # # # # # #

        # 0 , 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, ..., pn
        # x1, y1, w1, h1, C1, x2, y2, w2, h2, C2, p1, p2, ..., pn
        predicted_bbox_1 = y_pred[:, :, :, :5]
        
        predicted_bbox_2 = y_pred[:, :, :, 5:10]
        
        predicted_class_prob = y_pred[:, :, :, 10:]
        
        #print(predicted_bbox_1) #(?, 21, 21, 5)
        #print(predicted_bbox_2) #(?, 21, 21, 5)
        #print(predicted_class_prob) #(?, 21, 21, 10)
        
        true_box = y_true[:, :, :, :4]
        true_object_confidence = y_true[:, :, :, 4]
        true_class_prob = y_true[:, :, :, 5:]

        #print(true_box) #(?, 21, 21, 4)
        #print(true_object_confidence) #(?, 21, 21)
        #print(true_class_prob) #(?, 21, 21, 10)
        
        # # # # # # # # # # # # # # # # # # #
        # find responsible bboxes
        # # # # # # # # # # # # # # # # # # #        

        iou_bbox1 = calculate_IOU(predicted_bbox_1, true_box)
        #(?, 21, 21)
        iou_bbox2 = calculate_IOU(predicted_bbox_2, true_box)
        #(?, 21, 21)
        
        responsible_pred_bbox = tf.greater(iou_bbox1, iou_bbox2)
        #print(responsible_pred_bbox)#(?, 21, 21)
        responsible_pred_bbox = tf.cast(responsible_pred_bbox,tf.float32)
        responsible_pred_bbox = tf.expand_dims(responsible_pred_bbox, axis=3)
        #print(responsible_pred_bbox) #(?, 21, 21, 1)
        
        responsible_pred_bbox = tf.tile(responsible_pred_bbox, [1, 1, 1, 5])
        #print(responsible_pred_bbox)#(?, 21, 21, 5)
        
        #responsible_pred_bbox = tf.where(responsible_pred_bbox, predicted_bbox_1, predicted_bbox_2)
        responsible_pred_bbox = tf.multiply(predicted_bbox_1, responsible_pred_bbox) + tf.multiply(predicted_bbox_2, tf.ones_like(responsible_pred_bbox) - (responsible_pred_bbox))
        
        #print(responsible_pred_bbox)#(?, 21, 21, 5)
        
        # # # # # # # # # # # # # # # # # # #
        # x,y loss
        # # # # # # # # # # # # # # # # # # #
        
        
        # (x - x')^2 + (y - y')^2
        x_loss = tf.squared_difference(true_box[..., 0], responsible_pred_bbox[..., 0])
        #print(x_loss) #(?, 21, 21)
        y_loss = tf.squared_difference(true_box[..., 1], responsible_pred_bbox[..., 1])
        #print(y_loss) #(?, 21, 21)
        xy_loss = tf.add(x_loss, y_loss)

        # if the object is not present in the cell that the sum is zero
        #print(true_object_confidence)#(?, 21, 21)
        xy_loss = tf.multiply(xy_loss, true_object_confidence)#(?, 21, 21)
        
        xy_loss = self.lambda_coord * tf.reduce_sum(xy_loss)
        tf.losses.add_loss(xy_loss)
        #print(xy_loss)  # shape=()

        # # # # # # # # # # # # # # # # # # #
        # w,h loss
        # # # # # # # # # # # # # # # # # # #

        # (sqrt(w) - sqrt(w'))^2 + (sqrt(h) - sqrt(h'))^2
        #true_box = tf.Print(true_box, [true_box], message='true_box \t', summarize=1000)
        #responsible_pred_bbox = tf.Print(responsible_pred_bbox, [responsible_pred_bbox], message='responsible_pred_bbox \t', summarize=1000)
        w_loss = tf.squared_difference(tf.sqrt(true_box[..., 2] + 1e-10) , tf.sqrt(responsible_pred_bbox[..., 2] + 1e-10) )
        h_loss = tf.squared_difference(tf.sqrt(true_box[..., 3] + 1e-10) , tf.sqrt(responsible_pred_bbox[..., 3] + 1e-10) )
        wh_loss = tf.add(w_loss, h_loss)

        # if the object is not present in the cell that the sum is zero
        wh_loss =  tf.multiply(wh_loss, true_object_confidence)
        wh_loss = self.lambda_coord * tf.reduce_sum(wh_loss)
        tf.losses.add_loss(wh_loss)
        # # # # # # # # # # # # # # # # # # #
        # bbox confidence loss
        # # # # # # # # # # # # # # # # # # #

        object_loss = tf.squared_difference(true_object_confidence, responsible_pred_bbox[..., 4])
        object_loss = tf.reduce_sum(tf.multiply(object_loss, true_object_confidence))

        no_object_loss = tf.squared_difference(true_object_confidence, responsible_pred_bbox[..., 4])
        no_object_loss = tf.reduce_sum(self.lambda_noobj * tf.multiply(no_object_loss, tf.ones_like(true_object_confidence) - true_object_confidence))


        confidence_loss = tf.add(object_loss, no_object_loss)
        tf.losses.add_loss(confidence_loss)
        #print(confidence_loss)

        # # # # # # # # # # # # # # # # # # #
        # classification loss
        # # # # # # # # # # # # # # # # # # #
        classification_loss = tf.squared_difference(true_class_prob, predicted_class_prob)
        classification_loss = tf.reduce_sum(classification_loss, axis=3)
        classification_loss = tf.reduce_sum(tf.multiply(classification_loss, true_object_confidence))
        tf.losses.add_loss(classification_loss)
        #print(classification_loss)



        # # # # # # # # # # # # # # # # # # #
        # Total loss
        # # # # # # # # # # # # # # # # # # #
        loss = tf.losses.get_total_loss()
        #xy_loss ok
        
        #loss = wh_loss
        # # # # # # # # # # # # # # # # # # #
        # Debug Info
        # # # # # # # # # # # # # # # # # # #
        #loss_print = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
        '''
        loss = loss - loss
        #loss = tf.Print(loss, [xy_loss], message='Loss XY \t', summarize=1000)
        #loss = tf.Print(loss, [wh_loss], message='Loss WH \t', summarize=1000)
        #loss = tf.Print(loss, [confidence_loss], message='Loss Conf \t', summarize=1000)
        #loss = tf.Print(loss, [classification_loss], message='Loss Class \t', summarize=1000)
        
        #loss_1 = tf.losses.absolute_difference(y_true,y_pred)
        '''
        # loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
        return loss

    def build_yolo_model(self, output_layer_size):
        input_image = Input(shape=self.input_size)
        
        inception_model = InceptionV3(include_top=False,
                                      weights='imagenet',
                                      input_tensor=None,
                                      input_shape=self.input_size,
                                      pooling=None)
        
        inception_model.trainable = False

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
        x = Dense(units=output_layer_size, name='yolo_dense_2', activation='relu')(x)

        model = Model(inputs=input_image, outputs=x)

        return model


    def predict(self, image_path, model_path, output_file_path):

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        model = load_model(model_path, custom_objects={'custom_loss': self.custom_loss, 'custom_acc': self.custom_acc})


        # fake batch
        images = np.expand_dims(image, axis=0)

        prediction = model.predict(images)

        np.save(output_file_path, prediction)

        print('Result successfully saved')








