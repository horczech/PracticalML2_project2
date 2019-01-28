import tensorflow as tf


def calculate_IOU(bbox1, bbox2):
    """

    :param bbox1: (x,y,w,h)
    :param bbox2: (x,y,w,h)
    :return: iou
    """

    intersection_x = tf.minimum(
        bbox1[..., 0] + 0.5 * bbox1[..., 2], bbox2[..., 0] + 0.5 * bbox2[..., 2]) - tf.maximum(
        bbox1[..., 0] - 0.5 * bbox1[..., 2], bbox2[..., 0] - 0.5 * bbox2[..., 2])

    intersection_y = tf.minimum(
        bbox1[..., 1] + 0.5 * bbox1[..., 3], bbox2[..., 1] + 0.5 * bbox2[..., 3]) - tf.maximum(
        bbox1[..., 1] - 0.5 * bbox1[..., 3], bbox2[..., 1] - 0.5 * bbox2[..., 3])

    intersection_area = tf.multiply(tf.maximum(0.0, intersection_x), tf.maximum(0.0, intersection_y))

    union_area = tf.subtract(tf.multiply(bbox1[..., 2], bbox1[..., 3]) + tf.multiply(bbox2[..., 2], bbox2[..., 3]),
                             intersection_area)

    iou = tf.divide(intersection_area, union_area)

    return iou