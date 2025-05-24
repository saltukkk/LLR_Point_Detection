from typing import List, Tuple

import tensorflow.keras as keras
from PIL import Image

from detection_datatypes import DetectionROI, DetectionBox, Point
from yolov4 import YOLOv4

MODEL_NAME = "model_28.07.2022_3_classes_p6.h5"

YOLO_MODEL = YOLOv4.from_file(
    f"../models/{MODEL_NAME}",
    f"configs/{MODEL_NAME}/yolov4-p6.cfg",
    f"configs/{MODEL_NAME}/obj.names"
)


def detect_rois(pil_image: Image.Image) -> List[Tuple[DetectionROI, DetectionBox]]:
    """Detects regions of interest on the image and returns coordinates in a dictionary format"""
    global YOLO_MODEL

    image = keras.preprocessing.image.img_to_array(pil_image)

    prediction = YOLO_MODEL.predict_constant_height(image, prediction_height=1400, min_obj_score=0.6,
                                                    min_class_score=0.5,
                                                    iou_threshold=0.25, max_boxes=40, letterbox_resizing=True)

    detections_list = []


    for detection in prediction:
        name = YOLO_MODEL.names[detection.best_class]

        ROI = DetectionROI.FEMUR_HEAD
        if name == 'tibia_top':
            ROI = DetectionROI.KNEE
        elif name == 'tibia_bottom':
            ROI = DetectionROI.ANKLE

        detection_repr = DetectionBox(
            upper_left=Point(
                y=detection.box.upper_left.y, x=detection.box.upper_left.x
            ),
            bottom_right=Point(
                y=detection.box.bottom_right.y, x=detection.box.bottom_right.x
            ),
            confidence=detection.objectness
        )

        detections_list.append(
            (ROI, detection_repr)
        )


    return detections_list
