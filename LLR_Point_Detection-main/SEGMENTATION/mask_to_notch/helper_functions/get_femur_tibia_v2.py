from PIL import Image
import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes
import cv2

def calc_masks(file_path, is_left=True):
    if is_left:
        img, segs = get_img_and_masks(file_path, is_left=True)
        if img is None:
            raise ValueError("Invalid image file path provided.")
        femur_mask, tibia_mask = process_leg(img, segs)
    else:
        img, segs = get_img_and_masks(file_path, is_left=False)
        if img is None:
            raise ValueError("Invalid image file path provided.")
        femur_mask, tibia_mask = process_leg(img, segs)
    return img, femur_mask, tibia_mask


def prep_mask(mask, num_largest_areas=2, fill=True):
    """
    Process a binary mask to fill holes, extract the largest connected regions
    """
    if fill:
        mask = binary_fill_holes(mask)  # Fill small holes in the mask

    label_mask = label(mask)  # Label connected components in the mask
    rprops = regionprops(label_mask)
    rprops_sorted = sorted(rprops, reverse=True, key=lambda x: x.area)[:num_largest_areas]

    # Create a new mask with only the largest regions
    new_mask = np.zeros_like(mask)
    for prop in rprops_sorted:
        for pt in prop.coords:
            new_mask[pt[0], pt[1]] = 1

    return new_mask

def filter_connected_components(binary_mask, size_threshold=0.1):
    # Convert boolean mask to uint8 for OpenCV
    binary_mask = binary_mask.astype(np.uint8)

    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1:  # No components found
        return binary_mask  # Return the original empty mask

    # Get the areas of all components (ignoring background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    # Find the largest component's area
    largest_area = np.max(areas)
    
    # Define a threshold for keeping smaller components
    min_area = largest_area * size_threshold

    # Create a new mask keeping only valid components
    filtered_mask = np.zeros_like(binary_mask, dtype=bool)

    for i, area in enumerate(areas):
        label = i + 1  # Since label 0 is background
        if area >= min_area:
            filtered_mask[labels == label] = True  # Preserve this region

    return filtered_mask  # Returns a boolean mask

def process_leg(img, segs, thresh=0.5, fill_holes=True, num_largest_areas=2):
    """
    Process the input image and segmentation masks for the left leg
    to extract femur and tibia masks.
    """
    
    img = np.array(img)  # Convert PIL image to NumPy array

    # Threshold segmentation masks to get binary masks
    mask_femur = segs[1] > thresh  # Femur mask
    mask_tibia = segs[2] > thresh  # Tibia mask
    
    # Get the largest connected component in each mask
    # Careful, maybe not good for some bad segmentations!!!
    mask_femur = filter_connected_components(mask_femur)
    mask_tibia = filter_connected_components(mask_tibia)

    # Process the femur mask
    filled_mask_femur = prep_mask(
        mask_femur, num_largest_areas=num_largest_areas, fill=fill_holes
    )

    # Process the tibia mask
    filled_mask_tibia = prep_mask(
        mask_tibia, num_largest_areas=num_largest_areas, fill=fill_holes
    )

    return filled_mask_femur, filled_mask_tibia


def extract_masks_from_png(file_path, is_left=True):
    """
    Extracts binary masks for background, femur, and tibia from a PNG image
    with three unique colors (black, white, gray).
    """
    # Step 1: Load the image
    img = Image.open(file_path)
    img_array = np.array(img)  # Convert to a NumPy array

    # Step 2: Find unique colors in the image
    unique_colors = np.unique(img_array.reshape(-1, 4), axis=0)
    print(f"Unique colors in the image: {unique_colors}")

    # Step 3: Define color-to-label mapping
    # Manually assign unique colors to segmentation labels
    # Replace these colors with the actual ones from your image

    if is_left: # left leg
        color_to_label = {
            (0, 0, 0, 255): "background",  # Black
            (128, 128, 128, 255): "femur",  # Gray
            (255, 255, 255, 255): "tibia",  # White
        }
    else: # right leg
        color_to_label = {
            (0, 0, 0, 255): "background",  # Black
            (64, 64, 64, 255): "femur",  # Dark-Gray
            (192, 192, 192, 255): "tibia",  # Light-Gray
        }

    # Step 4: Create binary masks for each label
    masks = []
    for color, label in color_to_label.items():
        # Create a binary mask for the current color
        mask = np.all(img_array == np.array(color), axis=-1).astype(np.uint8)
        masks.append(mask)

    return masks

def get_img_and_masks(file_path, is_left=True):
    return Image.open(file_path), extract_masks_from_png(file_path, is_left=is_left)

# Example usage
if __name__ == "__main__":
    # Dummy inputs for demonstration
    file_path = "sol.png"
   #  print_image_matrix(file_path)

    img, masks = get_img_and_masks(file_path)
    femur_mask, tibia_mask = process_leg(img, masks)
    np.savetxt("right_tibia_mask.txt", tibia_mask, fmt='%d')
    np.savetxt("right_femur_mask.txt", femur_mask, fmt='%d')
    print("Femur and Tibia masks processed.")
