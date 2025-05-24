#!/usr/bin/python3

# This module loads and serves the darknet model as a sliding window algorithm without
# multiple inferences and without the need of additional weights files
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import configparser
import numpy as np
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import cmp_to_key

from typing import List


class MultiOrderedDict(OrderedDict):
    """ Multi Ordered Dict for config parser with duplicate sections """

    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super().__setitem__(key, value)

    @staticmethod
    def getlist(value):
        return value.split(os.linesep)

@dataclass
class YOLOPoint:
    y: float
    x: float

    def shift(self, dy, dx):
        self.y += dy
        self.x += dx

    def scale(self, sy, sx):
        self.y *= sy
        self.x *= sx

@dataclass
class YOLOBox:
    upper_left: YOLOPoint
    bottom_right: YOLOPoint

    def shift(self, dy, dx):
        self.upper_left.shift(dy, dx)
        self.bottom_right.shift(dy, dx)

    def scale(self, sy, sx):
        self.upper_left.scale(sy, sx)
        self.bottom_right.scale(sy, sx)

    def width(self):
        return self.bottom_right.x-self.upper_left.x

    def height(self):
        return self.bottom_right.y-self.upper_left.y

    def center_x(self):
        return (self.upper_left.x+self.bottom_right.x)/2

    def center_y(self):
        return (self.upper_left.y+self.bottom_right.y)/2

    def area(self):
        return self.width()*self.height()

    @staticmethod
    def box_intersection(box_1, box_2):
        """Returns intersection of two boxes"""
        intersection_upper_left = YOLOPoint(
            y=max(box_1.upper_left.y, box_2.upper_left.y),
            x=max(box_1.upper_left.x, box_2.upper_left.x)
        )
        intersection_bottom_right = YOLOPoint(
            y=min(box_1.bottom_right.y, box_2.bottom_right.y),
            x=min(box_1.bottom_right.x, box_2.bottom_right.x)
        )
        if intersection_bottom_right.y < intersection_upper_left.y or \
            intersection_bottom_right.x < intersection_upper_left.x:
            return YOLOBox(
                upper_left = YOLOPoint(0, 0),
                bottom_right = YOLOPoint(0, 0)
            )
        return YOLOBox(
            upper_left=intersection_upper_left,
            bottom_right=intersection_bottom_right
        )

    @staticmethod
    def box_outline(box_1, box_2):
        """Returns a minimum box that encompasses both boxes"""
        outline_upper_left = YOLOPoint(
            y=min(box_1.upper_left.y, box_2.upper_left.y),
            x=min(box_1.upper_left.x, box_2.upper_left.x)
        )
        outline_bottom_right = YOLOPoint(
            y=max(box_1.bottom_right.y, box_2.bottom_right.y),
            x=max(box_1.bottom_right.x, box_2.bottom_right.x)
        )
        return YOLOBox(
            upper_left=outline_upper_left,
            bottom_right=outline_bottom_right
        )


@dataclass
class YOLODetection:
    box: YOLOBox
    objectness: float
    class_prob: List[float]

    def shift(self, dy, dx):
        self.box.shift(dy, dx)

    def scale(self, sy, sx):
        self.box.scale(sy, sx)

    def __post__init__(self):
        # As in darknet source
        self.sort_class = -1
        self.best_class = -1

    def iou(self, other):
        I = YOLOBox.box_intersection(self.box, other.box).area()
        U = self.box.area()+other.box.area()-I

        if I == 0 or U == 0:
            return 0
        return I/U

    def diou_nms(self, other, beta1):
        outline = YOLOBox.box_outline(self.box, other.box)
        w = outline.bottom_right.x-outline.upper_left.x
        h = outline.bottom_right.y-outline.upper_left.y

        c = w * w + h * h

        iou = self.iou(other)

        if c == 0:
            return iou

        dx = (self.box.center_x()-other.box.center_x())
        dy = (self.box.center_y()-other.box.center_y())

        d = dx*dx + dy*dy
        u = tf.pow(d/c, beta1)
        diou_term = u # not really required, to be exact with darknet
        return iou-diou_term




    @staticmethod
    def diou_nms_comparator(detection_1, detection_2):
        # As in darknet source
        diff = 0
        if detection_2.sort_class >= 0:
            diff = detection_1.class_prob[detection_2.sort_class] - detection_2.class_prob[detection_2.sort_class]
        else:
            diff = detection_1.objectness - detection_2.objectness
        if diff < 0:
            return 1
        elif diff > 0:
            return -1
        return 0

class YOLOv4:
    """
    Represents a YOLOv4 model based on the AlexeyAB darknet implementation.
    Requires a saved keras-format (hdf5, .h5) saved weights format.

    Anchor settings are predefined for the YOLOv4 model.
    """

    def __init__(self, model, config_path, names_path):
        """
        Initialises the YOLOv4 model with the saved weights and
        classes information. Use factory methods for automated loading.
        :param model: A loaded keras model (keras.Model object)
        :param config_path: Path to the model's config file (.cfg)
        :param names_path: Path to the objects/classes names file (one name per line)
        """
        self.model = model

        # Read the class names from the name file
        # Currently the class names are unused by the model
        with open(names_path, "r") as names_file:
            names_lines = names_file.read().split("\n")
            names_lines = filter(lambda x: len(x.strip()) > 0, names_lines)
            names_lines = list(map(lambda x: x.strip(), names_lines))

            names = {idx: name for idx, name in enumerate(names_lines)}
            self.names = names

        config_parser = configparser.RawConfigParser(dict_type=MultiOrderedDict, strict=False,
                                                     converters={'list': MultiOrderedDict.getlist})
        config_parser.read(config_path)

        # Determine global model parameters
        self.model_width = int(config_parser.get("net", "width"))
        self.model_height = int(config_parser.get("net", "height"))
        self.model_conv_size = 64
        self.model_channels = int(config_parser.get("net", "channels"))
        self.model_n_classes = int(config_parser["yolo"]["classes"][0])
        # Determine duplicate parameters
        self.model_output_anchors = []
        self.model_output_mask = []
        self.model_scales = []
        self.model_new_coords = []
        for model_anchors, model_mask, scale, coords in zip(config_parser.getlist("yolo", "anchors"),
                                                            config_parser.getlist("yolo", "mask"),
                                                            config_parser.getlist("yolo", "scale_x_y"),
                                                            config_parser.getlist("yolo", "new_coords")):
            anchor_values = model_anchors.split(",")
            anchor_values = filter(lambda x: len(x.strip()) > 0, anchor_values)
            anchor_values = map(lambda x: int(x.strip()), anchor_values)
            mask_indices = model_mask.split(",")
            mask_indices = filter(lambda x: len(x.strip()) > 0, mask_indices)
            mask_indices = map(lambda x: int(x.strip()), mask_indices)

            scale = float(scale.strip())

            new_coords = bool(int(coords.strip()))

            self.model_output_anchors.insert(0, list(anchor_values))
            self.model_output_mask.insert(0, list(mask_indices))
            self.model_scales.insert(0, scale)
            self.model_new_coords.insert(0, new_coords)

    @classmethod
    def from_file(cls, path, config_path, names_path):
        """
        Loads the model from the model files.
        :param path: Path to the model's .h5 saved weights file.
        :param config_path: Path to the model's .cfg config file.
        :param names_path: Path to the model's .names file.
        :return: Instance of the model loaded from the supplied files.
        """
        custom_objects = {
            "mish": tfa.activations.mish
        }
        model = keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        return cls(model, config_path, names_path)

    def predict_constant_height(self, input_image, prediction_height=64 * 50,
                                min_class_score=0.1, max_boxes=50, iou_threshold=0.3,
                                min_obj_score=0.9, letterbox_resizing=True):
        """
        Predict bounding boxes for the image by resizing it to a constant height.
        The aspect ratio is kept so the width is scaled accordingly.
        :param input_image: input image. If converting the image to tf.constant fails, try loading it with
            keras.preprocessing.image.load_img.
        :param prediction_height: Height of the image on which the inference will be run. Image will be rescaled
            to this height while keeping aspect ratio.
        :param min_class_score: Minimum bounding box confidence for it to be returned to the result.
        :param max_boxes: Maximum number of bounding boxes in the result.
        :param iou_threshold: Intersect over Union threshold for two bounding boxes to be considered overlapping.
        :param min_obj_score: Minimum probability of an object being detected for the bounding box to be kept.
        :param letterbox_resizing: Whether to use letterbox resizing during inference.
        :return: List of tuples of the format:
            ((y1, x1, y2, x2), cls), where y, x represent absolute image coordinates of the top left (y1, x1) and
            bottom right (y2, x2) bounding box. The cls represents the object class index.
        """
        input_tensor = tf.constant(input_image)
        aspect_ratio = input_tensor.shape[1] / input_tensor.shape[0]
        prediction_width = int(aspect_ratio * prediction_height)

        model_dimensions, as_tensor = self._preprocess_image(input_image, prediction_height,
                                                             prediction_width,
                                                             letterbox_resizing)

        return self._prediction_core(as_tensor, input_tensor.shape[:2], model_dimensions,
                                     min_class_score=min_class_score, max_boxes=max_boxes, iou_threshold=iou_threshold,
                                     min_obj_score=min_obj_score, letterbox_resizing=True)

    def predict(self, input_image, min_class_score=0.9, max_boxes=50, iou_threshold=0.5, min_obj_score=0.9):
        """
        Predict bounding boxes for the image without resizing it or any further preprocessing.
        WARNING! For large images this might run for a long time. Use predict_constant_height to ensure
        comparable prediction times.
        :param input_image: input image. If converting the image to tf.constant fails, try loading it with
            keras.preprocessing.image.load_img..
        :param min_class_score: Minimum bounding box confidence for it to be returned to the result.
        :param max_boxes: Maximum number of bounding boxes in the result.
        :param iou_threshold: Intersect over Union threshold for two bounding boxes to be considered overlapping.
        :param min_obj_score: Minimum probability of an object being detected for the bounding box to be kept.
        :return: List of tuples of the format:
            ((y1, x1, y2, x2), cls), where y, x represent absolute image coordinates of the top left (y1, x1) and
            bottom right (y2, x2) bounding box. The cls represents the object class index.
        """
        model_dimensions, as_tensor = self._preprocess_image(input_image, *input_image.shape[:2])

        return self._prediction_core(as_tensor, tf.constant(input_image).shape[:2], model_dimensions,
                                     min_class_score=min_class_score,
                                     max_boxes=max_boxes, iou_threshold=iou_threshold,
                                     min_obj_score=min_obj_score)

    def _prediction_core(self, preprocessed_image, input_dimensions, model_dimensions, min_class_score=0.7,
                         max_boxes=50, iou_threshold=0.5, min_obj_score=0.8, letterbox_resizing=True):
        """
        Core of the prediction method. This method is responsible for creating a tensorflow-compliant
        batched representation of the image.
        :param preprocessed_image: Image for inference after preprocessing (already a tf.constant and appropriate size).
        :param input_dimensions: Dimensions of the input image before preprocessing (required for scaling the bounding
            boxes back to the original size). A (height, width) tuple.
        :param model_dimensions: Dimensions of the model input. A (height, width) tuple.
        :param min_class_score: Minimum bounding box confidence for it to be returned to the result.
        :param max_boxes: Maximum number of bounding boxes in the result.
        :param iou_threshold: Intersect over Union threshold for two bounding boxes to be considered overlapping.
        :param min_obj_score: Minimum probability of an object being detected for the bounding box to be kept.
        :param letterbox_resizing: If letterbox resizing was used to pad the iamge
        :return: List of tuples of the format:
            ((y1, x1, y2, x2), cls), where y, x represent absolute image coordinates of the top left (y1, x1) and
            bottom right (y2, x2) bounding box. The cls represents the object class index.
        """
        batched_image = tf.expand_dims(preprocessed_image, axis=0)

        model_prediction = self.model.predict(batched_image)

        bounding_boxes = self._parse_yolo_output(model_prediction, preprocessed_image.shape[:2],
                                                 min_class_score=min_class_score,
                                                 min_obj_score=min_obj_score)

        # Perform DIOU NMS sorting
        self._diou_nms_sort(bounding_boxes, iou_threshold)

        # Perform final threshold filtering
        bounding_boxes = self._boxes_best_filter(bounding_boxes, min_class_score)


        if not letterbox_resizing:
            self._rescale_bounding_boxes(bounding_boxes, input_dimensions)
            return bounding_boxes


        # If letterbox resizing was used, bounding boxes have to be corrected to source image
        # coordinate space
        letterbox_dy = -(preprocessed_image.shape[0]-model_dimensions[0])/(2*preprocessed_image.shape[0])
        letterbox_dx = -(preprocessed_image.shape[1]-model_dimensions[1])/(2*preprocessed_image.shape[1])

        s_y = preprocessed_image.shape[0]/model_dimensions[0]
        s_x = preprocessed_image.shape[1]/model_dimensions[1]

        for box in bounding_boxes:
            box.shift(letterbox_dy, letterbox_dx)
            box.scale(s_y, s_x)

        self._rescale_bounding_boxes(bounding_boxes,
                                            input_dimensions)
        return bounding_boxes

    def _boxes_best_filter(self, boxes, min_class_score):
        result = []
        for detection in boxes:
            best_class = tf.argmax(detection.class_prob).numpy()
            if detection.class_prob[best_class] >= min_class_score:
                detection.best_class = best_class
                result.append(detection)
        return result


    def _diou_nms_sort(self, boxes: List[YOLODetection], iou_threshold, beta1=0.6, nms_kind="diou"):
        assert nms_kind=="diou", "Only DIOU NMS is currently supported"

        total = len(boxes)

        k = total-1
        i = 0
        while i <= k:
            if boxes[i].objectness == 0:
                boxes[k], boxes[i] = boxes[i], boxes[k]
                k -= 1
                i -= 1
            i += 1

        total = k+1

        for k in range(0, self.model_n_classes):
            for box in boxes[:total]:
                box.sort_class = k
            boxes.sort(key=cmp_to_key(YOLODetection.diou_nms_comparator))
            for box_a_idx in range(total):
                if boxes[box_a_idx].class_prob[k] == 0:
                    continue
                for box_b_idx in range(box_a_idx+1, total):
                    if nms_kind == "diou":
                        if boxes[box_a_idx].diou_nms(boxes[box_b_idx], beta1) > iou_threshold:
                            boxes[box_b_idx].class_prob[k] = 0


    def _preprocess_image(self, image, height, width, letterbox_resizing=False):
        """
        Preprocesses the image to a target width while keeping the YOLO requirement
        for the image dimensions to be multiple of 32. Also takes care of
        reducing and adding dimensions for RGBA or grayscale images respectively.
        :param image: Input image
        :param height: Target image height
        :param width: Target image width
        :param letterbox_resizing: Whether to resize as a letterbox
        :return: (Target width rounded to 32, target height rounded to 32), preprocessed image.
        """

        def closest_multiple(number, multiple):
            rounded = round(number / multiple) * multiple
            if rounded == 0:
                rounded = multiple
            return int(rounded)

        as_tensor = tf.constant(image)

        if as_tensor.shape[2] == 1:  # Input is grayscale
            as_tensor = tf.repeat(as_tensor, repeats=3, axis=-1)

        if as_tensor.shape[2] == 4:  # Input has alpha channel
            as_tensor = as_tensor[:, :, :3]

        if tf.reduce_max(as_tensor) > 1:  # Rescale image
            as_tensor = tf.cast(as_tensor, tf.float32)
            as_tensor = as_tensor / tf.constant(255.0)

        target_height = closest_multiple(height, self.model_conv_size)
        target_width = closest_multiple(width, self.model_conv_size)

        if not letterbox_resizing:
            return (target_height, target_width), tf.image.resize(as_tensor, [target_height, target_width])

        return (target_height, target_width), tf.image.resize_with_pad(as_tensor-0.5, target_height,
                                                                       target_height)+0.5

    def _parse_yolo_output(self, model_prediction, image_shape, min_class_score=0.7, min_obj_score=0.8):
        """
        Concatenates the YOLO output from multiple prediction layers and converts the bounding boxes to
        a human-readable format.
        :param model_prediction: List of raw darknet predictions (values of the last layers).
        :param image_shape: Shape of the image on which the prediction was carried out.
        :param min_class_score: Minimum bounding box confidence for it to be returned to the result.
        :param max_boxes: Maximum number of bounding boxes in the result.
        :param iou_threshold: Intersect over Union threshold for two bounding boxes to be considered overlapping.
        :param min_obj_score: Minimum probability of an object being detected for the bounding box to be kept.
        :return: List of tuples of the format:
            ((y1, x1, y2, x2), cls), where y, x represent absolute image coordinates of the top left (y1, x1) and
            bottom right (y2, x2) bounding box. The cls represents the object class index. The coordinates
            are limited by the image_shape dimensions.
        """
        # Each of the yolo outputs predicts at a different scale, mask defines the bounding box indices
        result = []

        for layer_index, prediction in enumerate(model_prediction):
            # Extract anchors
            try:
                anchors = self._extract_anchors(
                    self.model_output_anchors[layer_index],
                    self.model_output_mask[layer_index]
                )
            except IndexError:
                raise IndexError(f"Anchors and anchor mask was not found for YOLO output layer {layer_index}. " + \
                                 f"This probably means the config file was not properly parsed.")

            scale = self.model_scales[layer_index]
            new_coords = self.model_new_coords[layer_index]
            grid_shape = prediction.shape[1:3]
            n_anchors = len(anchors)

            # In the input each of the results is represented as a class score,
            # box b_x, b_y, width and height, and an obj score,
            # thus the final dimension has shape n_classes+5
            per_anchor_output = tf.reshape(prediction, (-1, *grid_shape, n_anchors, self.model_n_classes + 5))

            box_xy, box_wh, obj_score, class_prob = self._parse_darknet_output(anchors, per_anchor_output, scale,
                                                                               new_coords)
            # Reshape the bounding boxes to the original image scale
            box_xy /= tf.reverse(tf.constant(grid_shape, tf.float32), axis=[0])
            # Reshape the bounding boxes to the original image scale
            box_wh /= tf.reverse(tf.constant(image_shape, tf.float32), axis=[0])

            # convert from  (x, y) & (w, h) to (y1, x1, y2, x2)
            top_left = tf.reverse(box_xy - box_wh/2, axis=[-1])  # - box_wh / 2
            bottom_right = tf.reverse(box_xy + box_wh/2, axis=[-1])  # + box_wh / 2

            boxes = tf.concat([top_left, bottom_right], axis=-1)

            # boxes, scores, classes = self._bounding_box_score_filter(boxes, obj_score, class_prob,
            #                                                          min_class_score=0,
            #                                                          min_obj_score=0)
            boxes, scores, classes = tf.reshape(boxes, [-1, 4]), tf.reshape(obj_score, [-1]), \
                                        tf.reshape(class_prob, [-1, self.model_n_classes])

            valid_obj = tf.where(scores >= min_obj_score)

            boxes, scores, classes = tf.gather(boxes, valid_obj, axis=0), tf.gather(scores, valid_obj, axis=0), \
                                     tf.gather(classes, valid_obj, axis=0)


            for box, score, class_ in zip(boxes, scores, classes):
                class_score = class_*score
                filtered_class_score = tf.where(class_score > min_class_score, class_score, 0)
                result.append(
                    YOLODetection(
                        box = YOLOBox(
                            upper_left=YOLOPoint(y=box[0,0], x=box[0,1]),
                            bottom_right=YOLOPoint(y=box[0,2], x=box[0,3])
                        ),
                        objectness=tf.squeeze(score).numpy(),
                        class_prob=tf.squeeze(filtered_class_score).numpy().tolist()
                    )
                )

        return result

    def _parse_darknet_output(self, anchors, per_anchor_output, scale, new_coords=False):
        """
        Parses the raw darknet output and converts the bounding boxes to a human-readable format.
        :param anchors: List of the anchor dimensions for the given layer (anchor_w, anchor_h)
        :param per_anchor_output: Output of the model for each of the anchors.
        :param scale: scale of the output position.
        :param new_coords: Whether to use new coordinate activation functions (for scaled Yolo variants).
        :return: (box_xy, box_wh, obj_score, class_prob), where:
            box_xy is a matrix of the bounding box xy coordinates for each bounding box and each of the anchors,
                the matrix has the following dimensions: (1, grid_position_y, grid_position_x, anchor_index, coords).
                The coordinates are represented as displacement [0.,1.0) from the grind top-left corner.
            box_wh represents the width and height of the box as a relative size of the anchor box. It has the same
                dimension box_xy has.
            obj_score represents the objectness score for each of the bounding boxes -
                probability it contains an object.
                The dimensions are (1, grid_position_y, grid_position_x, anchor_index, 1)
            class_prob represents class probabilities for each of the bounding boxes.
                The dimensions are (1, grid_position_y, grid_position_x, anchor_index, class_index)
        """
        # To get bounding box offsets it has to be passed via the sigmoid function first
        # Mostly sourced from
        # https://gist.github.com/zzxvictor/200f40b436541cf7af365ae5d5a80a90#file-extractcoord-py
        n_anchors = len(anchors)

        anchor_indices = self._generate_image_offsets(per_anchor_output.shape[1:3])
        anchor_indices = tf.cast(anchor_indices, per_anchor_output.dtype)

        anchors = tf.cast(tf.reshape(anchors, (1, 1, 1, n_anchors, 2)), anchor_indices.dtype)

        if new_coords:
            box_xy = per_anchor_output[..., :2]
            box_xy = box_xy*scale - (scale-1.0)/2.0
            box_xy = box_xy + anchor_indices

            box_wh = per_anchor_output[..., 2:4]
            box_wh = tf.math.pow(box_wh * 2.0, 2)
            box_wh = box_wh * anchors
        else:
            box_xy = tf.nn.sigmoid(per_anchor_output[..., :2])
            box_xy = per_anchor_output[..., :2]
            box_xy = box_xy*scale
            box_xy = box_xy + anchor_indices

            print("WARNING! Using outdated coordination schema. It probably doesn't match" + \
                  "with darknet and WILL cause invalid predictions.")
            box_wh = per_anchor_output[..., 2:4]
            box_wh = tf.math.exp(box_wh)
            box_wh = box_wh * anchors

        obj_score = per_anchor_output[..., 4:5]
        class_prob = per_anchor_output[..., 5:]

        return box_xy, box_wh, obj_score, class_prob

    @staticmethod
    def _extract_anchors(anchors, anchor_mask):
        """
        Given the requested anchor indices, extracts their width and height from the config file.
        :param anchors: list of anchors from which to extact
        :param anchor_mask: List of anchor indices for extraction
        :return: List of anchor dimensions - (width, height) tuples.
        """
        output_anchors = []
        for anchor_index in anchor_mask:
            output_anchors.append(
                (anchors[anchor_index * 2 ], anchors[anchor_index * 2 + 1])
            )
        return tf.constant(output_anchors, tf.float32)

    @staticmethod
    def _generate_image_offsets(grid_shape):
        """
        Given the requested grid shape, generates grid index offsets for each grid cell.
        :param grid_shape: Grid shape (width, height).
        :return: Matrix of grid index displacements for the whole grid in the following format:
            (1, grid_index_y, grid_index_x, 1, coords=2 - y or x)
        """
        # In darknet width dimension is first in the shape
        horizontal_index = tf.reshape(tf.range(start=0, limit=grid_shape[1]), (1, grid_shape[1]))
        horizontal_index = tf.tile(horizontal_index, [grid_shape[0], 1])
        vertical_index = tf.reshape(tf.range(start=0, limit=grid_shape[0]), (grid_shape[0], 1))
        vertical_index = tf.tile(vertical_index, [1, grid_shape[1]])
        index = tf.stack([horizontal_index, vertical_index], axis=-1)
        index = tf.reshape(index, shape=(1, *grid_shape, 1, 2))
        return index

    @staticmethod
    def _non_max_suppress(boxes, scores, classes, max_boxes=50, iou_threshold=0.5,
                          score_threshold=0.8):
        """
        Performs non-max suppression on the input data and flattens the output to a simple list.
        :param boxes: Matrix of bounding boxes, rows represent (y1, x1, y2, x2) coordinates.
        :param scores: Nx1 matrix of bounding box class*objectness scores.
                (so score is already multiplied by the class score).
        :param classes: Nxn_classes matrix of bounding box class scores
        :param max_boxes: Maximum number of boxes remaining after suppression.
        :param iou_threshold: Intersect over Union threshold for two bounding boxes to be considered overlapping
        :return: (boxes, scores, classes), where each tuple element is a list of corresponding
            bounding boxes, scores, and class scores.
        """
        indices = tf.image.non_max_suppression(boxes, scores, max_boxes,
                                               iou_threshold=iou_threshold,
                                               score_threshold=score_threshold)
        boxes = tf.gather(boxes, indices)
        scores = tf.gather(scores, indices)
        classes = tf.gather(classes, indices)
        return boxes, scores, classes

    @staticmethod
    def _bounding_box_score_filter(boxes, scores, classes, min_class_score=0.0, min_obj_score=0.0):
        """
        Filters the bounding boxes
        :param boxes: Matrix of bounding boxes, rows represent (y1, x1, y2, x2) coordinates.
        :param scores: Nx1 matrix of bounding box objectness scores.
        :param classes: Nxn_classes matrix of bounding box class scores
        :param min_class_score: Minimum class_score*objectness_score for an object to be considered detected.
        :return: (boxes, scores, classes), a tuple representing the same boxes as input, but only
            if their objectness_score*max_class_score is >= min_class_score.
        """
        # filter results by class score
        box_score = scores * classes
        box_class = tf.argmax(box_score, axis=-1)
        #box_score = tf.math.reduce_max(classes, axis=-1)
        box_score = tf.math.reduce_max(scores*classes, axis=-1)
        mask = (box_score >= min_class_score) & (tf.reshape(scores, box_score.shape) >= min_obj_score)
        # return boxes, scores, box_class
        return tf.boolean_mask(boxes, mask), tf.boolean_mask(scores, mask), tf.boolean_mask(box_class, mask)

    @staticmethod
    def _rescale_bounding_boxes(bounding_boxes, input_dimensions):
        """
        This method rescales the bounding boxes from one image size to another.
        :param bounding_boxes: List of bounding boxes (y1, x1, y2, x2) to be rescaled.
        :param input_dimensions: Dimensions of the input image, a (height, width) tuple.
        :return: List of the bounding boxes with coordinates rescaled so that their relative
            position is the same in the input and processing dimensions.
        """
        y_ratio = input_dimensions[0]
        x_ratio = input_dimensions[1]

        for box in bounding_boxes:
            box.scale(y_ratio, x_ratio)

    def _experimental_predict_sliding_window(self, input_image, window_width="model", window_height="model", stride=256,
                                             min_class_score=0.9, max_boxes=50, iou_threshold=0.3):
        if window_width == "model":
            window_width = self.model_width
        if window_height == "model":
            window_height = self.model_height

        _, as_tensor = self._preprocess_image(input_image, *input_image.shape[:2])

        as_batch = tf.expand_dims(as_tensor, axis=0)

        patches = tf.image.extract_patches(
            images=as_batch,
            sizes=[1, window_height, window_width, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches_rows = patches.shape[1]
        patches_cols = patches.shape[2]

        predictions = []

        for row in range(patches_rows):
            for col in range(patches_cols):
                patch = tf.reshape(patches[0, row, col, :], [window_height, window_width, as_tensor.shape[2]])
                patch_predictions = self._prediction_core(patch, patch.shape[:2], patch.shape[:2],
                                                          min_class_score=min_class_score, max_boxes=max_boxes,
                                                          iou_threshold=iou_threshold)
                patch_y = row * stride
                patch_x = col * stride

                for ((y1, x1, y2, x2), cls) in patch_predictions:
                    predictions.append(
                        ((y1 + patch_y, x1 + patch_x, y2 + patch_y, x2 + patch_x), cls)
                    )

        return predictions
