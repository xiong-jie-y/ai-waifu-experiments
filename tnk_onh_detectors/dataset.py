import os
import sys
sys.path.append(os.path.expanduser("~/gitrepos/models/research"))

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import open3d

import image_utils
import bounding_box_utils as bd_utils
import numpy as np

from typing import *

import slam_map

from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2


def _validate_label_map(label_map):
  """Checks if a label map is valid.
  Args:
    label_map: StringIntLabelMap to validate.
  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 1:
      raise ValueError('Label map ids should be >= 1.')

def save_label_map_dict(path, label_map_dict):
  """Saves label map proto from a dictionary of label names to id.
  Args:
    path: path to save StringIntLabelMap proto text file.
    label_map_dict: dictionary of label names to id.
  """
  label_map = string_int_label_map_pb2.StringIntLabelMap()
  for name, item_id in sorted(label_map_dict.items(), key=lambda x: x[1]):
    label_map.item.add(name=name, id=item_id)

  _validate_label_map(label_map)
  label_map_string = text_format.MessageToString(label_map)
  with tf.gfile.GFile(path, 'wb') as fid:
    fid.write(label_map_string)

def create_tf_example(image_id, image, bounding_box_list, class_name_id_map):
    calib = slam_map.get_intrinsic()
    encoded_image_data, width, height = image_utils.convert_from_open3d_color_image_to_png(image)
    filename = f"{image_id}.png".encode('utf-8')
    image_format = b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    for label, bounding_box in bounding_box_list:
        bounding_box = bd_utils.get_2d_min_max_bouding_box(bounding_box, calib)
        if bounding_box['min'][0] < 0 or bounding_box['min'][1] < 0 \
            or bounding_box['max'][0] >= width or bounding_box['max'][1] >= height:
            continue

        xmins.append(bounding_box['min'][0] / width)
        xmaxs.append(bounding_box['max'][0] / width)
        ymins.append(bounding_box['min'][1] / height)
        ymaxs.append(bounding_box['max'][1] / height)
        classes_text.append(label.encode('utf-8'))

    if len(classes_text) == 0:
        return None

    classes = [class_name_id_map[cls_name.decode('utf-8')] for cls_name in classes_text]

    # print(width)
    # print(xmins)
    # print(width, height)
    # print(classes)
    # print(classes_text)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_records_from(data_tuple_lists, class_name_id_map, output_path, folder_name):
    train_dir = os.path.join(output_path, folder_name)
    os.makedirs(train_dir, exist_ok=True)
    writer = tf.python_io.TFRecordWriter(os.path.join(train_dir, "data.tfrecord"))
    num_examples = 0
    for i, (image, bounding_box_list) in enumerate(data_tuple_lists):
        tf_example = create_tf_example(i, image.color, bounding_box_list, class_name_id_map)
        if tf_example is None:
            continue
        writer.write(tf_example.SerializeToString())
        num_examples += 1
    writer.close()
    print(f"num_examples: {num_examples}")

import random
def create_object_detection_api_tf_record(
    images: List[open3d.geometry.RGBDImage],
    bounding_boxes: List[List[Tuple[str, open3d.geometry.OrientedBoundingBox]]], 
    output_path: str):
    data_tuple_lists = list(zip(images, bounding_boxes))
    sep_point = int(len(data_tuple_lists)*0.9)
    random.shuffle(data_tuple_lists)
    data_tuple_lists_train = data_tuple_lists[:sep_point]
    data_tuple_lists_val = data_tuple_lists[sep_point:]

    print(len(data_tuple_lists_val))

    class_names_set = set([])
    for _, data_tuple_list in data_tuple_lists:
        for label, _ in data_tuple_list:
            class_names_set.add(label)

    class_name_id_map = {}
    for i, cls_uniq in enumerate(class_names_set):
        # ID should start from 1.
        class_name_id_map[cls_uniq] = i + 1

    print(class_name_id_map)
    map_path = os.path.join(output_path, "tf_label_map.pbtxt")
    os.makedirs(output_path, exist_ok=True)
    save_label_map_dict(map_path, class_name_id_map)

    create_tf_records_from(data_tuple_lists_train, class_name_id_map, output_path, "train")
    create_tf_records_from(data_tuple_lists_val, class_name_id_map, output_path, "val")