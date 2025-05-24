import numpy as np
from skimage.morphology import binary_erosion
from helper_functions.bresenham_slope import bres


def get_cont(mask):
    mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8)
    eroded_mask = binary_erosion(mask)
    diff_mask = mask - eroded_mask

    return diff_mask


def get_contpt(mask):
    diff_mask = get_cont(mask)
    return np.nonzero(diff_mask)


def get_side_contour_pts(mask, y: int):
    nonzero = np.nonzero(mask[y])[0]
    if len(nonzero) == 0:  # Handle empty row
        return None, None
    return nonzero.min(), nonzero.max()


def calc_disc(mask, y: int):
    side_pts = get_side_contour_pts(mask, y)
    if side_pts[0] is None:
        return None
    return np.nonzero(1 - mask[y, side_pts[0] : side_pts[1]])[0] + side_pts[0]

def calc_large_gap(mask, y: int, gaps_prev, largest_g):
    
    row = mask[y]
    zero_indices = np.where(row == 0)[0]
    if len(zero_indices) == 0:
        return []

    gaps = []
    start = zero_indices[0]
    for i in range(1, len(zero_indices)):
        if zero_indices[i] != zero_indices[i - 1] + 1:
            gaps.append((start, zero_indices[i - 1]))
            start = zero_indices[i]
    gaps.append((start, zero_indices[-1]))

    # Filter out gaps that start or end at the boundary
    non_boundary_gaps = [
        gap for gap in gaps if gap[0] != 0 and gap[1] != len(mask[y]) - 1
    ]

    # Find the largest non-boundary gap
    if non_boundary_gaps:
        largest_gap = max(non_boundary_gaps, key=lambda gap: gap[1] - gap[0])
    else:
        largest_gap = (-1, -1)
    old_strt, old_end = largest_g
    new_strt, new_end = largest_gap
    if largest_gap != (-1,-1) and new_end - new_strt > old_end - old_strt:
        return non_boundary_gaps, largest_gap, True
    else:
        return gaps_prev, largest_g, False
    # if len(gaps) > 2 and len(gaps) > len(gaps_prev):
    #     return gaps, largest_gap
    # else:
    #     return gaps, largest_g

    

def get_lowest_mask_pt(mask):
    pts = get_contpt(mask)
    indices = np.argsort(pts[0])

    return pts[0][indices[-1]], pts[1][indices[-1]]


def get_high_pt(mask):
    pts = get_contpt(mask)
    indices = np.argsort(pts[0])
    return pts[0][indices[0]], pts[1][indices[0]]


def shrink_points(mask, start_pt, end_pt):
    points_on_line = bres([start_pt], end_pt, -1)

    shrinked_start, shrinked_end = None, None

    for pt in points_on_line:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_start = pt
            break

    for pt in points_on_line[::-1]:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_end = pt
            break

    return shrinked_start, shrinked_end
