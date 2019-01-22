import json
import os

from model import YOLO
from constants import CONFIG_FILE
from pathlib2 import Path
import glob



def parse_annotation_file(path):
    objects = []

    with open(str(path)) as fp:
        lines = fp.readlines()

    for line in lines:
        object = {}

        class_id, x_center, y_center, width, height = line.strip().split(' ')

        object['class_id'] = int(class_id)
        object['x_center'] = float(x_center)
        object['y_center'] = float(y_center)
        object['width'] = float(width)
        object['height'] = float(height)

        objects.append(object)

    return objects


def parse_input_data(image_folder, annotation_folder, annotation_extension, image_extension):
    data_infos = []

    if not os.path.exists(str(image_folder)) or not os.path.exists(str(annotation_folder)):
        raise ValueError('Entered file path does not exist! Entered Paths: ' + str(image_folder) + " and " + str(annotation_folder))

    image_names = glob.glob(str(image_folder) + '/*' + image_extension)
    if len(image_names) == 0:
        raise ValueError('No images found')

    for image_name in image_names:
        annotation_path = annotation_folder.joinpath(Path(image_name).stem + annotation_extension)

        data_info = {}
        data_info["image_path"] = image_name
        data_info["objects"] = parse_annotation_file(annotation_path)

        data_infos.append(data_info)

    return data_infos


def _main_():
    with open(CONFIG_FILE) as config_buffer:
        config = json.loads(config_buffer.read())

    ################################
    # Load data info
    ################################
    train_data_infos = parse_input_data(image_folder=Path(config['train']['train_images_folder']),
                                        annotation_folder=Path(config['train']['train_annotations_folder']),
                                        annotation_extension=config['train']['annotations_format_extension'],
                                        image_extension=config['train']['image_format_extension'])

    validation_data_infos = parse_input_data(image_folder=Path(config['train']['validation_images_folder']),
                                             annotation_folder=Path(config['train']['validation_annotations_folder']),
                                             annotation_extension=config['train']['annotations_format_extension'],
                                             image_extension=config['train']['image_format_extension'])

    ################################
    # Make and train model
    ################################
    yolo = YOLO(input_size          = tuple(config['model']['input_size']),
                grid_size           = int(config['model']['grid_size']),
                bbox_count          = int(config['model']['bboxes_per_grid_cell']),
                classes             = config['model']['class_names'],
                lambda_coord        = config['model']['lambda_coord'],
                lambda_noobj        = config['model']['lambda_noobj'],
                bbox_params         = config['model']['bbox_params'])

    yolo.train_gen(training_infos       = train_data_infos,
                   validation_infos     = validation_data_infos,
                   save_weights_path    = config['train']['trained_weights_path'],
                   batch_size           = config['train']['batch_size'],
                   nb_epochs            = config['train']['nb_epochs'],
                   learning_rate        = config['train']['learning_rate'])


if __name__ == '__main__':
    _main_()