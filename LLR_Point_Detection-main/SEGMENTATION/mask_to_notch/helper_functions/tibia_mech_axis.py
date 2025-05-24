import numpy as np
from numpy.polynomial import Polynomial
from helper_functions.ankle import determine_mid_of_ankle
from helper_functions.knee import determine_mid_of_knee
from helper_functions.utils import determine_mid_of_joint_x, get_top_width, get_min_y_per_x

def get_knee_level(mask, percentage_mid=0.5, percentage_side=0.15):

    min_x, max_x = get_top_width(mask)
    possible_indices = np.arange(np.floor(min_x), np.ceil(max_x))

    mid = int(len(possible_indices)/2)

    num_pts_mid = int(len(possible_indices) * percentage_mid)
    num_pts_side = int(len(possible_indices)*percentage_side)
    lower_mid = int(mid - num_pts_mid/2)
    upper_mid = int(mid + num_pts_mid/2)

    x_vals = np.concatenate(
        [possible_indices[num_pts_side:lower_mid],
         possible_indices[upper_mid:-num_pts_side]])

    y_vals = np.array([get_min_y_per_x(mask, _x) for _x in x_vals])

    line = Polynomial.fit(
        x_vals,
        y_vals, 1,
        domain=[x_vals.min(), x_vals.max()])

    return line(possible_indices), possible_indices

def calc_mechanical_axis(
    mask,
    percentage_mid=0.3,
    percentage_side=0.2,
    leave_out_percentage=0.15,
):
    knee_level_femur = get_knee_level(mask)
    mech_axis_top = determine_mid_of_knee(
        mask,
        femur_level=knee_level_femur,
        percentage_mid=percentage_mid,
        percentage_side=percentage_side,
    )
    mech_axis_bottom = determine_mid_of_ankle(
        mask, leave_out_percentage=leave_out_percentage
    )

    return mech_axis_top, mech_axis_bottom
