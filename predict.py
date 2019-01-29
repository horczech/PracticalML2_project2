from constants import CONFIG_FILE
import json
from keras.models import Model, load_model
import cv2
import os
from model import YOLO

if __name__ == '__main__':

    with open(CONFIG_FILE) as config_buffer:
        config = json.loads(config_buffer.read())


    image_path = os.path.expanduser(config["predict"]["image_path"])
    model_path = os.path.expanduser(config["predict"]["model_path"])
    output_file_path = os.path.expanduser(config["predict"]["output"])


    print('>>>>> Creating YOLO object')
    yolo = YOLO(input_size=tuple(config['model']['input_size']),
                grid_size=int(config['model']['grid_size']),
                bbox_count=int(config['model']['bboxes_per_grid_cell']),
                classes=config['model']['class_names'],
                lambda_coord=config['model']['lambda_coord'],
                lambda_noobj=config['model']['lambda_noobj'],
                bbox_params=config['model']['bbox_params'])

    if os.path.isfile(image_path) and os.path.isfile(model_path):
      yolo.predict(image_path, model_path,output_file_path)

    else:
        print('Path to image or model does not exist...')
