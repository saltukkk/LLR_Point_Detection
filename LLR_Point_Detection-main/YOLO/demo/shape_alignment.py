from typing import List, Tuple, Dict
import logging
import pickle
import pathlib
import platform

from detection_datatypes import DetectionROI, DetectionBox, Point, DetectionResult, DetectionLeg, CalculationResult, \
    Line

from skimage import util
import numpy as np
from skimage.filters import unsharp_mask, meijering, sobel_h
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.filters.rank import enhance_contrast
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.morphology import disk, closing, remove_small_objects
from skimage.measure import regionprops, label

from PIL import Image

import menpo


def detect_head_center(image, bbox, margin):
    """ Detect fermoral head center and its radius """
    image = np.array(image)
    img = image[bbox[0][0] - margin:bbox[0][1] + margin, bbox[1][0] - margin:bbox[1][1] + margin]
    img = util.img_as_ubyte(img)
    img = equalize_adapthist(img, clip_limit=0.03)
    img = unsharp_mask(img, radius=5, amount=115)
    img = util.img_as_ubyte(img)
    img = enhance_contrast(img, disk(7))
    edges = canny(img, sigma=5)
    max_radius = max(int(img.shape[0] / 2), int(img.shape[1] / 2))
    hough_radii = np.arange(int(0.65 * max_radius), int(0.95 * max_radius), 3)
    hough_res = hough_circle(edges, hough_radii)
    _, center_y, center_x, radius = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    x = center_x[0] + bbox[0][0] - margin
    y = center_y[0] + bbox[1][0] - margin
    return Point(y, x), radius


def identify_legs(pair, img_shape):
    if len(pair) == 0:
        leg_L = None
        leg_R = None
    elif len(pair) == 1:
        leg = pair[0]
        if leg['x1'] < img_shape[1] / 2:
            leg_L = None
            leg_R = ((leg['y1'], leg['y2']), (leg['x1'], leg['x2']))
        else:
            leg_R = None
            leg_L = ((leg['y1'], leg['y2']), (leg['x1'], leg['x2']))
    elif len(pair) == 2:
        leg_1 = pair[0]
        leg_2 = pair[1]
        if leg_1['x1'] > leg_2['x1']:
            leg_L = ((leg_1['y1'], leg_1['y2']), (leg_1['x1'], leg_1['x2']))
            leg_R = ((leg_2['y1'], leg_2['y2']), (leg_2['x1'], leg_2['x2']))
        else:
            leg_R = ((leg_1['y1'], leg_1['y2']), (leg_1['x1'], leg_1['x2']))
            leg_L = ((leg_2['y1'], leg_2['y2']), (leg_2['x1'], leg_2['x2']))
    return (leg_L, leg_R)


def align_ankle(image, bbox, crop_margin_y=(-15, 25)):
    """ Crop to ankle joint using meijering filter """
    image = np.array(image)
    img = meijering(image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]])
    y = np.argmax(np.sum(img, axis=1))
    min_lim = max(0, int(y) + crop_margin_y[0])
    max_lim = int(y + crop_margin_y[1])
    to_crop = (bbox[0][0] + min_lim, bbox[0][0] + max_lim)
    return to_crop, bbox[1]


def detect_ankle_points(image, crop_points):
    """ Returns inner and outer ankle point """
    image = np.array(image)
    image = sobel_h(image[crop_points[0][0]:crop_points[0][1], crop_points[1][0]:crop_points[1][1]])
    image = image > 0.03
    image = remove_small_objects(image, 15)
    image = closing(image, np.ones((5, 3)))
    image = remove_small_objects(image, 35)
    image = closing(image, np.ones((5, 3)))
    image = closing(image, np.ones((9, 19)))
    areas = [region.area for region in regionprops(label(image))]
    image = remove_small_objects(image, max(areas) - 100)
    start = np.where(np.any(image, axis=0))[0][0]
    end = np.where(np.any(image, axis=0))[0][-1]
    point1y = np.where(np.any(image[:, start:start + 5], axis=1))[0][0]
    point1x = np.where(np.any(image[point1y:point1y + 1, start:start + 5], axis=0))[0][0]
    point2y = np.where(np.any(image[:, end - 5:end], axis=1))[0][0]
    point2x = np.where(np.any(image[point2y:point2y + 1, end - 5:end], axis=0))[0][0]
    return [Point(point1x + start + crop_points[1][0], point1y + crop_points[0][0]),
            Point(point2x + end - 5 + crop_points[1][0], point2y + crop_points[0][0])]


def move_initial_shape_to_bbox(reference_shape, bbox):
    initial_shape = []
    x_points = []
    y_points = []
    for point in reference_shape.points:
        x_points.append(point[1])
        y_points.append(point[0])

    shape_center = (min(x_points) + max(x_points)) // 2
    bbox_center = (bbox[1][0] + bbox[1][1]) // 2

    for x, y in zip(x_points, y_points):
        x = x + bbox_center - shape_center
        y = y + bbox[0][0] - min(y_points)
        initial_shape.append([y, x])

    initial_shape = menpo.shape.PointCloud(initial_shape)
    return initial_shape


def detect_knee_points(image, bbox, model_path):
    """ Returns inner and outer knee point """

    print("Hello")
    
    idx1 = -2
    idx2 = -1
    
    print(model_path)
    
    image_menpo = menpo.image.Image(equalize_hist(util.img_as_float(np.array(image))))

    fitter = pickle.load(open(model_path, "rb"))
    reference_shape = fitter.reference_shape
    initial_shape = move_initial_shape_to_bbox(reference_shape, bbox)
    result = fitter.fit_from_shape(image_menpo, initial_shape, max_iters=100)

    ax = int(result.final_shape.points[idx1][0])
    ay = int(result.final_shape.points[idx1][1])
    bx = int(result.final_shape.points[idx2][0])
    by = int(result.final_shape.points[idx2][1])
    points = ((bx, by), (ax, ay))
    return [Point(points[0][1], points[0][0]), Point(points[1][1], points[1][0])]


def calculate_osteo_point(knee_inner, knee_outer, left_leg):
    """ Calculate osteotomy point """

    Y_PROP = 0.18
    X_PROP = 0.14

    knee_width = abs(knee_inner.x - knee_outer.x)
    y = int(knee_outer.y + knee_width * Y_PROP)
    x_displacement = knee_width * X_PROP

    if left_leg:
        x = knee_outer.x - x_displacement
    else:
        x = knee_outer.x + x_displacement

    return Point(x, y)


def calculate_angle(femur_head: Point,
                    knee_outer: Point,
                    knee_inner: Point,
                    ost_point: Point,
                    ankle_outer: Point,
                    ankle_inner: Point) -> CalculationResult:
    F_POINT_RATIO = 0.625
    # Fujisawa point calculation
    knee_line = Line.from_points(knee_outer, knee_inner)
    right_leg = knee_outer.x < knee_inner.x
    if right_leg:
        f_x = knee_inner.x - F_POINT_RATIO * np.abs(knee_outer.x - knee_inner.x)
    else:
        f_x = knee_inner.x + F_POINT_RATIO * np.abs(knee_outer.x - knee_inner.x)
    f_y = knee_line.get_y(f_x)
    f_point = Point(f_x, f_y)

    # Head-Fujisawa Line calculation
    f_line = Line.from_points(femur_head, f_point)

    # Ankle center point calculation
    a_x = (ankle_outer.x + ankle_inner.x) / 2
    a_y = (ankle_outer.y + ankle_inner.y) / 2
    a_point = Point(a_x, a_y)

    # Ankle line calculation
    a_line = Line.from_points(ankle_outer, ankle_inner)

    # Head-Fujisawa line and ankle line cross point calculation
    c_point = f_line.cross_point(a_line)

    # Angle calculation
    angle = get_angle(a_point, ost_point, c_point)

    return CalculationResult(f_point, f_line, a_point, a_line, c_point, angle)


def get_angle(a: Point, b: Point, c: Point) -> float:
    """ Returns angle between ba and bc lines. """

    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def detect_points(pil_image: Image.Image, detected_rois: List[Tuple[DetectionROI, DetectionBox]]) -> Dict[
    DetectionLeg, DetectionResult]:

    print("ABCD")
    
    MODEL_PATHS = ["../models/aam_full_h_59_left.p", "../models/aam_full_h_59_right.p"]
    
    print(MODEL_PATHS)
    
    plt = platform.system()
    if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    if plt == 'Darwin': pathlib.WindowsPath = pathlib.PosixPath        

    ankles = []
    knees = []
    femur_heads = []
    for roi in detected_rois:
        if roi[0] == DetectionROI.ANKLE:
            ankles.append({'y1': int(roi[1].upper_left.y.numpy()), 'x1': int(roi[1].upper_left.x.numpy()),
                           'y2': int(roi[1].bottom_right.y.numpy()), 'x2': int(roi[1].bottom_right.x.numpy())})
        if roi[0] == DetectionROI.KNEE:
            knees.append({'y1': int(roi[1].upper_left.y.numpy()), 'x1': int(roi[1].upper_left.x.numpy()),
                          'y2': int(roi[1].bottom_right.y.numpy()), 'x2': int(roi[1].bottom_right.x.numpy())})
        if roi[0] == DetectionROI.FEMUR_HEAD:
            femur_heads.append({'y1': int(roi[1].upper_left.y.numpy()), 'x1': int(roi[1].upper_left.x.numpy()),
                                'y2': int(roi[1].bottom_right.y.numpy()), 'x2': int(roi[1].bottom_right.x.numpy())})

    img_shape = pil_image.size

    knees = identify_legs(knees, img_shape)
    femur_heads = identify_legs(femur_heads, img_shape)
    ankles = identify_legs(ankles, img_shape)

    results = []
    for left_leg in [True, False]:
        points = {}
        bbox_idx = 0 if left_leg else 1
        head_point, head_radius = detect_head_center(pil_image, femur_heads[bbox_idx], margin=50)
        points['femur_head'] = head_point
        ankle = align_ankle(pil_image, ankles[bbox_idx])
        ankle_points_result = detect_ankle_points(pil_image, ankle)
        points['ankle_inner'] = ankle_points_result[int(left_leg)]
        points['ankle_outer'] = ankle_points_result[int(not left_leg)]

        knee_outer, knee_inner = detect_knee_points(pil_image, knees[bbox_idx], MODEL_PATHS[bbox_idx])
        osteo_point = calculate_osteo_point(knee_inner, knee_outer, left_leg=left_leg)
        points['knee_inner'] = knee_inner
        points['knee_outer'] = knee_outer
        points['ost_point'] = osteo_point
        points['correction_angle_in_deg'] = round(calculate_angle(**points).angle, 2)
        points['femur_head_radius'] = head_radius[0]

        results.append(points)

    return {
        DetectionLeg.LEFT: DetectionResult(**results[0]),
        DetectionLeg.RIGHT: DetectionResult(**results[1])
    }
