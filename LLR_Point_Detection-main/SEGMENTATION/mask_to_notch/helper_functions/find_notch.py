import numpy as np
from helper_functions.mask_utils import get_contpt, calc_disc, get_lowest_mask_pt, calc_large_gap
from copy import copy

def determine_min_y(mask, percentage=0.1):
    contour_pts = get_contpt(mask)

    num_pts = int(len(contour_pts[0]) * percentage)

    sorted_y = np.sort(contour_pts[0])
    return sorted_y[-num_pts]

# yeni başlandı:
#
# def get_two_side_contour_pts(mask, y, x):
#     """
#     Returns the nearest nonzero columns to the left and right of (y,x) in 'mask'.
#     If none exist on a side, returns None for that side.
#     """
#     nonzero = np.nonzero(mask[y])[0]
#     if len(nonzero) == 0:
#         return None, None  # No nonzero columns in this row

#     # Candidates to the left are those columns <= x
#     left_candidates = nonzero[nonzero <= x]
#     # Candidates to the right are those columns >= x
#     right_candidates = nonzero[nonzero >= x]

#     left = left_candidates.max() if len(left_candidates) else None
#     right = right_candidates.min() if len(right_candidates) else None

#     return left, right

# def find_notch(mask):
#     min_y = determine_min_y(mask, percentage=1.0)

#     height, width = mask.shape

#     found_second_hill = False
#     ends_short = 0
#     for y in range(min_y, 0, -1):
#         row = mask[y, :]  # Current row
#         row_diff = np.diff(row)  # Find transitions
#         starts = np.where(row_diff == 1)[0] + 1
#         ends = np.where(row_diff == -1)[0]

#         # Collect blobs >= 50 pixels
#         valid_blobs = []
#         for s, e in zip(starts, ends):
#             if (e - s + 1) >= 50:
#                 valid_blobs.append((s, e))
        
#         (s1,e1) = valid_blobs[0]
        
#         if len(valid_blobs) == 2:
#             found_second_hill = True

        
#         if found_second_hill and len(valid_blobs) == 1:
#             return y, e1

#     return min_y, width//2


## number 2
# def find_notch(mask, min_y=None, percentage=0.1):
#     if min_y is None:
#         min_y = determine_min_y(mask, percentage)
#     y_float, _ = get_lowest_mask_pt(mask)

#     y = int(y_float)

#     large_g = (-1,-1)
#     gaps = []

#     while y >= min_y:
#         gaps, large_g = calc_large_gap(mask, y, gaps, large_g)

#         y -= 1

#     srt, end = large_g
#     pts = np.nonzero(mask[:, int((srt + end) / 2)])
#     indices = np.argsort(pts[0])

#     notch = pts[0][indices[-1]], int((srt + end) / 2)
#     print(notch,  pts[0][indices[-1]], int((srt + end) / 2), indices[-1])
#     return notch

# number 1,75: classic mid point (a+b)/2
# def find_notch(mask, min_y=None, percentage=0.1):
#     if min_y is None:
#         min_y = determine_min_y(mask, percentage)
#     y_float, x_float = get_lowest_mask_pt(mask)

#     x = int(x_float)
#     y = int(y_float)

#     large_g = (-1,-1)
#     gaps = []

#     second_plato_x = None
#     second_plato_y = None
#     while y >= min_y:
#         gaps, large_g, found = calc_large_gap(mask, y, gaps, large_g)
#         if found:
#             second_plato_y = copy(y)
#         y -= 1

#     srt, end = large_g
#     print(large_g)
#     a = min(x,srt,end)
#     b = max(x,srt,end)

#     p1_pts = np.nonzero(mask[:, a])
#     p1_indices = np.argsort(p1_pts[0])
    
#     p2_pts = np.nonzero(mask[:, b])
#     p2_indices = np.argsort(p2_pts[0])

#     notch_pts = np.nonzero(mask[:, int((a + b) / 2)])
#     notch_indices = np.argsort(pts[0])

#     notch = notch_pts[0][notch_indices[-1]], int((a + b) / 2)
#     p1 = p1_pts[0][p1_indices[-1]], a
#     p2 = p2_pts[0][p2_indices[-1]], b

#     # print(notch,  pts[0][indices[-1]], int((srt + end) / 2), indices[-1])
#     return [notch, p1, p2]


# number 1,5: binary search for optimization
# def find_notch(mask, min_y=None, percentage=0.1):
#     if min_y is None:
#         min_y = determine_min_y(mask, percentage)
#     y_float, x_float = get_lowest_mask_pt(mask)

#     x = int(x_float)
#     y = int(y_float)

#     large_g = (-1,-1)
#     gaps = []

#     second_plato_x = None
#     second_plato_y = None
#     while y >= min_y:
#         gaps, large_g, found = calc_large_gap(mask, y, gaps, large_g)
#         if found:
#             second_plato_y = copy(y)
#         y -= 1

#     srt, end = large_g
#     print(large_g)
#     a = min(x,srt,end)
#     b = max(x,srt,end)

#     p1_pts = np.nonzero(mask[:, a])
#     p1_indices = np.argsort(p1_pts[0])
    
#     p2_pts = np.nonzero(mask[:, b])
#     p2_indices = np.argsort(p2_pts[0])
#     #futher optimization, binary search
#     l = a
#     h = b
#     l_val = p1_pts[0][p1_indices[-1]]
#     h_val = p2_pts[0][p2_indices[-1]]
#     for i in range(20):
#         mid = (h + l) // 2
#         temp_pts = np.nonzero(mask[:, mid])
#         temp_indices = np.argsort(temp_pts[0])

#         if len(temp_indices) > 0:
#             mid_val = temp_pts[0][temp_indices[-1]]
#             if mid_val < l_val:
#                 l = mid
#                 l_val = mid_val
#             elif mid_val < h_val:
#                 h = mid
#                 h_val = mid_val
#             else:
#                 break
#         else:
#             raise ValueError("No point found in the middle of the gap")

#     notch = h_val, h
#     p1 = p1_pts[0][p1_indices[-1]], a
#     p2 = p2_pts[0][p2_indices[-1]], b

#     # print(notch,  pts[0][indices[-1]], int((srt + end) / 2), indices[-1])
#     return [notch, p1, p2]


# number 1,15
# def find_notch(mask, min_y=None, percentage=0.1, degree=2):
#     if min_y is None:
#         min_y = determine_min_y(mask, percentage)
#     y_float, x_float = get_lowest_mask_pt(mask)

#     x = int(x_float)
#     y = int(y_float)

#     large_g = (-1,-1)
#     gaps = []

#     second_plato_x = None
#     second_plato_y = None
#     while y >= min_y:
#         gaps, large_g, found = calc_large_gap(mask, y, gaps, large_g)
#         if found:
#             second_plato_y = copy(y)
#         y -= 1

#     srt, end = large_g
#     print(large_g)
#     a = min(x,srt,end)
#     b = max(x,srt,end)

#     values = np.zeros(b - a, dtype=int)
#     for i in range(a, b):
#         pts = np.nonzero(mask[:, i])
#         indices = np.argsort(pts[0])
#         if len(indices) > 0:
#             values[i-a] = pts[0][indices[-1]]
#         else:
#             values[i-a] = -1  # Assign a default value if no points are found

    
#     def fit_parabola(arr):
#         x = np.arange(len(arr))
#         coeffs = np.polyfit(x, arr, degree)  # Fit a polynomial with 'degree' (2 is a parabola) 
#         a, b, c = coeffs
#         min_x = -b / (2 * a)  # Vertex of the parabola
#         return round(min_x) if 0 <= min_x < len(arr) else None  # Ensure index is valid

#     min_index = fit_parabola(values)

#     notch_x = min_index + a
#     notch_y = values[min_index]
#     p1_pts = np.nonzero(mask[:, a])
#     p1_indices = np.argsort(p1_pts[0])
#     p2_pts = np.nonzero(mask[:, b])
#     p2_indices = np.argsort(p2_pts[0])
#     p1 = p1_pts[0][p1_indices[-1]], a
#     p2 = p2_pts[0][p2_indices[-1]], b

#     return [(notch_y, notch_x), p1, p2]

# number 1,15
def find_notch(mask, min_y=None, percentage=0.1, degree=2):
    if min_y is None:
        min_y = determine_min_y(mask, percentage)
    y_float, x_float = get_lowest_mask_pt(mask)

    x = int(x_float)
    y = int(y_float)

    large_g = (-1,-1)
    gaps = []

    second_plato_x = None
    second_plato_y = None
    while y >= min_y:
        gaps, large_g, found = calc_large_gap(mask, y, gaps, large_g)
        if found:
            second_plato_y = copy(y)
        y -= 1

    srt, end = large_g
    print(large_g)
    a = min(x,srt,end)
    b = max(x,srt,end)

    min_index = -1
    min_val = 1000000
    values = np.zeros(b - a, dtype=int)
    for i in range(a, b):
        pts = np.nonzero(mask[:, i])
        indices = np.argsort(pts[0])
        if len(indices) > 0:
            values[i-a] = pts[0][indices[-1]]
            if values[i-a] < min_val:
                min_val = values[i-a]
                min_index = i
        else:
            values[i-a] = -1  # Assign a default value if no points are found

    notch_x = min_index
    notch_y = min_val
    p1_pts = np.nonzero(mask[:, a])
    p1_indices = np.argsort(p1_pts[0])
    p2_pts = np.nonzero(mask[:, b])
    p2_indices = np.argsort(p2_pts[0])
    p1 = p1_pts[0][p1_indices[-1]], a
    p2 = p2_pts[0][p2_indices[-1]], b

    return [(notch_y, notch_x), p1, p2]


# def find_notch(mask, min_y=None, percentage=0.1):
#     if min_y is None:
#         min_y = determine_min_y(mask, percentage)
#     y_float, _ = get_lowest_mask_pt(mask)

#     y = int(y_float)

#     allowed = False
#     notch = None, None

#     while y >= min_y:
#         disc = calc_disc(mask, wy)

#         if isinstance(disc, np.ndarray) and len(disc):
#             allowed = True

#         else:
#             if allowed:
#                 new_disc = calc_disc(mask, y + 1)
#                 notch = (y + 1, new_disc[int(len(new_disc) / 2)])
#                 # break
                
#                 allowed = False
#             # bir de böyle dene:
#             # allowed = False

#         y -= 1

#     return notch
