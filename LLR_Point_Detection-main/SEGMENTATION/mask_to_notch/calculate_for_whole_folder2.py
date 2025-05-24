import os
import cv2
import numpy as np
import multiprocessing
# from calculate_HKA2 import calculate_hka
# from calculate_AMA2 import calculate_ama
# from helper_functions.visualize_axis import visualize_lines_v2
from helper_functions.find_notch import find_notch
from helper_functions.get_femur_tibia_v2 import calc_masks
from PIL import Image, ImageDraw

def standardize_image_format(image_path, output_path):
    """
    Convert an image to a standardized RGBA format with consistent alpha values.
    
    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the standardized image.
    """
    # Open the image
    img = Image.open(image_path).convert("RGBA")
    
    # Standardize the alpha channel (set all non-transparent pixels to fully opaque)
    data = np.array(img)
    data[..., 3] = 255  # Set alpha channel to 255 (fully opaque)
    
    # Create a new image with the modified data
    standardized_img = Image.fromarray(data, mode="RGBA")
    
    # Save the standardized image
    standardized_img.save(output_path)
    print(f"Standardized image saved to {output_path}")

# Define the input and output directories
input_dir = '../bone_seg_nnunet_main/postprocessed'  # Replace with your input folder path
output_dir = '../../OUTPUT/OUTPUT_SEGMENTATION/'
output_dir_left = os.path.join(output_dir, './left')  # Replace with your output folder path for left leg
output_dir_right = os.path.join(output_dir, './right')  # Replace with your output folder path for right leg
# output_dir_boxes = os.path.join(output_dir, './boxes')  # Replace with your output folder path for images with bounding boxes
output_dir_results = os.path.join(output_dir, './results')  # Replace with your output folder path for results
test_dir = os.path.join(output_dir, './test')  # Replace with your output folder path for results

# Create output directories if they don't exist
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)
# os.makedirs(output_dir_boxes, exist_ok=True)
os.makedirs(output_dir_results, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Define pixel values for left and right legs
left_leg_values = [128, 255]  # Gray and White
right_leg_values = [64, 192]  # Dark-Gray and Light-Gray

# Function to add margin with black cover
def add_margin_with_black_cover(x, y, w, h, margin, img):
    y_margin = int(h * margin)
    x_margin = y_margin  # Fix left and right margin to the top margin
    new_h = h + 2 * y_margin
    new_w = w + 2 * x_margin
    new_img = np.zeros((new_h, new_w), dtype=img.dtype)
    new_img[y_margin:y_margin + h, x_margin:x_margin + w] = img[y:y + h, x:x + w]
    return new_img, x_margin, y_margin

def process_image(filename):
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Image not found at path: {image_path}")
        return

    # Convert the image to grayscale if it has an alpha channel
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create masks for left and right legs
    left_leg_mask = np.isin(image, left_leg_values).astype(np.uint8)
    right_leg_mask = np.isin(image, right_leg_values).astype(np.uint8)

    # Find contours for left leg
    contours_left, _ = cv2.findContours(left_leg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours for right leg
    contours_right, _ = cv2.findContours(right_leg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding box for left leg
    if contours_left:
        all_points = np.vstack(contours_left)
        x_left, y_left, w_left, h_left = cv2.boundingRect(all_points)
        left_leg_cropped, x_margin_left, y_margin_left = add_margin_with_black_cover(x_left, y_left, w_left, h_left, 0.1, image)
        left_output_path = os.path.join(output_dir_left, filename[:-4] + '_left_leg_cropped.png')
        cv2.imwrite(left_output_path, left_leg_cropped)
        print(f"Left leg cropped and saved as '{left_output_path}'")
    else:
        print(f"No contour found for left leg in image: {filename}")
        return

    # Get bounding box for right leg
    if contours_right:
        all_points = np.vstack(contours_right)
        x_right, y_right, w_right, h_right = cv2.boundingRect(all_points)
        right_leg_cropped, x_margin_right, y_margin_right = add_margin_with_black_cover(x_right, y_right, w_right, h_right, 0.1, image)
        right_output_path = os.path.join(output_dir_right, filename[:-4] + '_right_leg_cropped.png')
        cv2.imwrite(right_output_path, right_leg_cropped)
        print(f"Right leg cropped and saved as '{right_output_path}'")
    else:
        print(f"No contour found for right leg in image: {filename}")
        return

    # Optionally visualize the bounding boxes
    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # if contours_left:
    #     cv2.rectangle(output_image, (x_left, y_left), (x_left + w_left, y_left + h_left), (0, 255, 0), 2)
    # if contours_right:
    #     cv2.rectangle(output_image, (x_right, y_right), (x_right + w_right, y_right + h_right), (255, 0, 0), 2)
    # output_with_boxes_path = os.path.join(output_dir_boxes, filename[:-4] + '_output_with_boxes.png')
    # cv2.imwrite(output_with_boxes_path, output_image)
    # print(f"Bounding boxes visualized and saved as '{output_with_boxes_path}'")

    # Calculate angles and visualize
    sol_path = left_output_path
    sag_path = right_output_path
    standardize_image_format(sol_path, sol_path)
    standardize_image_format(sag_path, sag_path)
    # img_size = (1024, 256)

    # sol_hka_angle, sol_hka_vis_pack = calculate_hka(sol_path, is_left=True)
    # sol_ama_angle, sol_ama_vis_pack = calculate_ama(sol_path, is_left=True)
    # print(f"Calculated AMA Left: {sol_ama_angle:.2f} degrees")
    # print(f"Calculated HKA Left: {sol_hka_angle:.2f} degrees")

    # sag_hka_angle, sag_hka_vis_pack = calculate_hka(sag_path, is_left=False)
    # sag_ama_angle, sag_ama_vis_pack = calculate_ama(sag_path, is_left=False)
    # print(f"Calculated AMA Right: {sag_ama_angle:.2f} degrees")
    # print(f"Calculated HKA Right: {sag_hka_angle:.2f} degrees")

    # visualize_lines_v2(mask1=sol_hka_vis_pack['full_mask'], axis_points_list1=[
    #     sol_hka_vis_pack['femur_mechanical_axis'], sol_hka_vis_pack['tibia_mechanical_axis'], sol_ama_vis_pack['anatomical_axis']
    #     ], 
    #     mask2=sag_hka_vis_pack['full_mask'], axis_points_list2=[
    #     sag_hka_vis_pack['femur_mechanical_axis'], sag_hka_vis_pack['tibia_mechanical_axis'], sag_ama_vis_pack['anatomical_axis'],
    # ], output_file=os.path.join(output_dir_results, filename[:-4] + '_visualization.png'))

    # Find and record the notch coordinates
    img, left_leg_femur_mask, left_leg_tibia_mask = calc_masks(sol_path, is_left=True)
    img, right_leg_femur_mask, right_leg_tibia_mask = calc_masks(sag_path, is_left=False)
    left_tibia_plato = find_notch(left_leg_femur_mask)
    left_notch = left_tibia_plato[0]
    left_p1 = left_tibia_plato[1]
    left_p2 = left_tibia_plato[2]

    right_tibia_plato = find_notch(right_leg_femur_mask)
    right_notch = right_tibia_plato[0]
    right_p1 = right_tibia_plato[1]
    right_p2 = right_tibia_plato[2]


    # Save the results
    with open(os.path.join(output_dir_results, filename[:-4] + '_results.txt'), 'w') as f:
        # f.write(f"Calculated AMA Left: {sol_ama_angle:.2f} degrees\n")
        # f.write(f"Calculated HKA Left: {sol_hka_angle:.2f} degrees\n")
        # f.write(f"Calculated AMA Right: {sag_ama_angle:.2f} degrees\n")
        # f.write(f"Calculated HKA Right: {sag_hka_angle:.2f} degrees\n")

        if left_notch[0] is not None:
            left_notch_global = (int(left_notch[0] + y_left - y_margin_left), int(left_notch[1] + x_left - x_margin_left))
            left_p1_global = (int(left_p1[0] + y_left - y_margin_left), int(left_p1[1] + x_left - x_margin_left))
            left_p2_global = (int(left_p2[0] + y_left - y_margin_left), int(left_p2[1] + x_left - x_margin_left))
            f.write(f"Left Notch Coordinates: {left_notch_global}\n")
            f.write(f"Left P1 Coordinates: {left_p1_global}\n")
            f.write(f"Left P2 Coordinates: {left_p2_global}\n")
            print(f"Left Notch Coordinates: {left_notch_global}")
            print(f"Left P1 Coordinates: {left_p1_global}")
            print(f"Left P2 Coordinates: {left_p2_global}")

        if right_notch[0] is not None:
            right_notch_global = (int(right_notch[0] + y_right - y_margin_right), int(right_notch[1] + x_right - x_margin_right))
            right_p1_global = (int(right_p1[0] + y_right - y_margin_right), int(right_p1[1] + x_right - x_margin_right))
            right_p2_global = (int(right_p2[0] + y_right - y_margin_right), int(right_p2[1] + x_right - x_margin_right))
            f.write(f"Right Notch Coordinates: {right_notch_global}\n")
            f.write(f"Right P1 Coordinates: {right_p1_global}\n")
            f.write(f"Right P2 Coordinates: {right_p2_global}\n")
            print(f"Right Notch Coordinates: {right_notch_global}")

    # Visualization of notch etc on segmentation mask
    draw_notch(input_dir, filename, [left_notch_global, left_p1_global, left_p2_global, right_notch_global, right_p1_global, right_p2_global])

def draw_notch(input_dir, filename, pixels):
    
    # Load an existing image
    path = os.path.join(input_dir, filename)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)

    # # Define pixel coordinates
    # pixels = [
    # (4197, 840),   # Left Notch Coordinates
    # (4261, 986),   # Left P1 Coordinates
    # # (4251, -102),  # Left P2 Coordinates
    # (4189, 2154),  # Right Notch Coordinates
    # (4253, 2003),  # Right P1 Coordinates
    # (4244, 1265)   # Right P2 Coordinates
    # ]

    # Scatter the pixels on the image with filled circles
    radius = 5  # Radius of the circle
    for y, x in pixels:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")

    # Save the modified image
    path = os.path.join(test_dir, filename)
    image.save(path)
    print(f"Image saved as {filename}.png")

def calculate_for_whole_folder():
    # image 70_0 and 71_0 are problematic. 70 has singular regions. 71 is very problematic.
    file_list = [f for f in os.listdir(input_dir) if f != '70_0.png' and f != '71_0.png' and f.endswith('.png')]
    num_workers = min(multiprocessing.cpu_count(), len(file_list))  # Limit to available CPUs
    print(f"Using {num_workers} parallel workers")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_image, file_list)
    # for f in file_list:
    #     process_image(f)
    # process_image("41_0.png")

if __name__ == "__main__":
    calculate_for_whole_folder()
