import tensorflow as tf
import numpy as np

GRID_SIZE = 7
CLASSES_COUNT = 10
BBOX_PARAMETER_COUNT = 5
BBOX_COUNT = 2

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

cell_param_count = BBOX_COUNT * BBOX_PARAMETER_COUNT + CLASSES_COUNT
vector_size = GRID_SIZE * GRID_SIZE * cell_param_count

pedicted_shape = (GRID_SIZE, GRID_SIZE, cell_param_count)
training_shape = (GRID_SIZE, GRID_SIZE, CLASSES_COUNT + BBOX_PARAMETER_COUNT)

print('cell_param_count: ' + str(cell_param_count))
print('vector_size: ' + str(vector_size))
print('pedicted_shape: ' + str(pedicted_shape))
print('training_shape: ' + str(training_shape))
print('\n\n')


# ToDo: Dont forget to lambda constant

def create_bbox(x, y, w, h, box_conf, class_id):
    # (x, y, w, h, box confidence score, class probabilities)
    class_probabilities = np.zeros(shape=CLASSES_COUNT)
    class_probabilities[class_id] = 1
    bbox_params = np.array([x, y, w, h, box_conf])
    bbox_params = np.append(bbox_params, class_probabilities)

    return bbox_params


def calculate_IOU(bbox1, bbox2):
    """

    :param bbox1: (x,y,w,h)
    :param bbox2: (x,y,w,h)
    :return: iou
    """

    intersection_x = tf.minimum(
        bbox1[:, :, 0] + 0.5 * bbox1[:, :, 2], bbox2[:, :, 0] + 0.5 * bbox2[:, :, 2]) - tf.maximum(
        bbox1[:, :, 0] - 0.5 * bbox1[:, :, 2], bbox2[:, :, 0] - 0.5 * bbox2[:, :, 2])

    intersection_y = tf.minimum(
        bbox1[:, :, 1] + 0.5 * bbox1[:, :, 3], bbox2[:, :, 1] + 0.5 * bbox2[:, :, 3]) - tf.maximum(
        bbox1[:, :, 1] - 0.5 * bbox1[:, :, 3], bbox2[:, :, 1] - 0.5 * bbox2[:, :, 3])

    intersection_area = tf.multiply(tf.maximum(0.0, intersection_x), tf.maximum(0.0, intersection_y))

    union_area = tf.subtract(tf.multiply(bbox1[:, :, 2], bbox1[:, :, 3]) + tf.multiply(bbox2[:, :, 2], bbox2[:, :, 3]),
                             intersection_area)

    iou = tf.divide(intersection_area, union_area)

    return iou


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, training_shape)
    y_pred = tf.reshape(y_pred, pedicted_shape)

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
    xy_loss = LAMBDA_COORD * tf.reduce_sum(xy_loss)

    # # # # # # # # # # # # # # # # # # #
    # w,h loss
    # # # # # # # # # # # # # # # # # # #

    # (sqrt(w) - sqrt(w'))^2 + (sqrt(h) - sqrt(h'))^2
    w_loss = tf.squared_difference(tf.sqrt(true_box[:, :, 2]), tf.sqrt(responsible_pred_bbox[:, :, 2]))
    h_loss = tf.squared_difference(tf.sqrt(true_box[:, :, 3]), tf.sqrt(responsible_pred_bbox[:, :, 3]))
    wh_loss = w_loss + h_loss

    # if the object is not present in the cell that the sum is zero
    wh_loss = wh_loss * true_object_confidence
    wh_loss = LAMBDA_COORD * tf.reduce_sum(wh_loss)

    # # # # # # # # # # # # # # # # # # #
    # bbox confidence loss
    # # # # # # # # # # # # # # # # # # #

    object_loss = tf.squared_difference(true_object_confidence, responsible_pred_bbox[:, :, 4])
    object_loss = tf.reduce_sum(object_loss * true_object_confidence)

    no_object_loss = tf.squared_difference(true_object_confidence, responsible_pred_bbox[:, :, 4])
    no_object_loss = tf.reduce_sum(LAMBDA_NOOBJ * tf.multiply(no_object_loss, 1 - true_object_confidence))

    confidence_loss = object_loss + no_object_loss

    # # # # # # # # # # # # # # # # # # #
    # classification loss
    # # # # # # # # # # # # # # # # # # #
    classification_loss = tf.squared_difference(true_class_prob, predicted_class_prob)
    classification_loss = tf.reduce_sum(classification_loss, axis=2)
    classification_loss = tf.reduce_sum(classification_loss * true_object_confidence)

    return xy_loss + wh_loss + confidence_loss + classification_loss


def run():
    # # # # # # # # # # # # # # # # # # #
    # Create y_pred and y_true
    # # # # # # # # # # # # # # # # # # #

    # create true vector
    y_true_np = np.zeros(training_shape)

    object_1 = create_bbox(x=0.5, y=0.5, w=0.3, h=0.3, box_conf=1.0, class_id=9)
    object_2 = create_bbox(x=0.1, y=0.1, w=0.1, h=0.1, box_conf=1.0, class_id=1)

    y_true_np[3, 3, :] = object_1
    # y_true_np[6, 6, :] = object_2

    print('TRUE bbox example:')
    print(y_true_np[3, 3, :])

    y_true_np = y_true_np.flatten()
    # y_true_np = np.reshape(y_true_np, (training_shape))

    # create predicted vector
    y_pred_np = np.zeros(pedicted_shape)

    object_1 = create_bbox(x=0.4, y=0.4, w=0.2, h=0.2, box_conf=0.8, class_id=9)
    object_1_1 = create_bbox(x=0.3, y=0.3, w=0.1, h=0.1, box_conf=0.7, class_id=5)

    y_pred_np[3, 3, :] = np.append([0.3, 0.3, 0.1, 0.1, 0.7], object_1)

    print('PREDICTED bbox example:')
    print(y_pred_np[3, 3, :])

    y_pred_np = y_pred_np.flatten()
    # y_true_np = np.reshape(y_true_np, (training_shape))

    # convert to tensor
    y_true = tf.convert_to_tensor(y_true_np, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred_np, dtype=tf.float32)

    print('\n\nRESULT:\n')
    with tf.Session() as sess:
        print(sess.run(loss_function(y_true=y_true, y_pred=y_pred)))


if __name__ == '__main__':
    run()
