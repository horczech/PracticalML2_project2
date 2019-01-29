import numpy as np
import keras
import cv2
import math
import copy

from imgaug import augmenters as iaa


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

        ######################################################
        #augmentors by https://github.com/aleju/imgaug
        ######################################################

        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )



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

            img_aug, target_aug = self.aug_image(img, data_info['objects'])

            # img_aug = self.aug_pipe.augment_image(img)
            # img_aug = img

            img_aug = cv2.resize(img_aug, (self.X_shape[0], self.X_shape[1]))
            img_aug = cv2.normalize(img_aug, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # parse labels (result)
            target_aug = self.parse_object_data(target_aug)

            # store sample
            X[i,] = img_aug

            # Store class
            y[i] = target_aug

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

    def aug_image(self, image, target_dics):
        result_target_dics = []

        h, w, c = image.shape

        # scale the image
        scale = np.random.uniform() / 10. + 1.
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # translate the image
        max_offx_relative = (scale - 1.)
        max_offy_relative = (scale - 1.)

        offx_relative = int(np.random.uniform() * max_offx_relative)
        offy_relative = int(np.random.uniform() * max_offy_relative)

        offx_abs = offx_relative * w
        offy_abs = offy_relative * h

        image = image[offy_abs: (offy_abs + h), offx_abs: (offx_abs + w)]

        ### flip the image
        flip = np.random.binomial(1, .5)
        if flip > 0.5:
            image = cv2.flip(image, 1)

        image = self.aug_pipe.augment_image(image)

        for bbox in target_dics:

            # fix object's position and size
            bbox['x_center'] = bbox['x_center'] - offx_relative
            bbox['y_center'] = bbox['y_center'] - offy_relative
            bbox['width'] = bbox['width'] * scale
            bbox['height'] = bbox['height'] * scale

            if flip > 0.5:
                bbox['x_center'] = 1 - (bbox['x_center'] + bbox['width'])

            if bbox['x_center'] < 0 or bbox['x_center'] > 1 or bbox['y_center'] < 0 or bbox['y_center'] > 1 :
                continue
            else:
                result_target_dics.append(bbox)

        return image, result_target_dics
