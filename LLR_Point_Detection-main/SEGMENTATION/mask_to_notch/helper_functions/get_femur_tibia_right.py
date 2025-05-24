from PIL import Image
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import resize
from scipy.ndimage.morphology import binary_fill_holes

def prep_mask(mask, num_largest_areas=2, fill=True, img_size=None):
    """
    Process a binary mask to fill holes, extract the largest connected regions,
    and optionally resize the mask.
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

    if img_size is not None:
        new_mask = resize(new_mask, img_size, preserve_range=True)

    return new_mask

def process_right_leg(img, segs, img_size=(1024, 256), thresh=0.5, fill_holes=True, num_largest_areas=2):
    """
    Process the input image and segmentation masks for the left leg
    to extract femur and tibia masks.
    """
    # Resize the input image
    img = np.array(img)  # Convert PIL image to NumPy array
    img = resize(img, img_size, preserve_range=True).astype(np.uint8)

    # Threshold segmentation masks to get binary masks
    mask_femur = segs[1] > thresh  # Femur mask
    mask_tibia = segs[2] > thresh  # Tibia mask

    # Process the femur mask
    resized_mask_femur = prep_mask(
        mask_femur, num_largest_areas=num_largest_areas, fill=fill_holes, img_size=img_size
    )

    # Process the tibia mask
    resized_mask_tibia = prep_mask(
        mask_tibia, num_largest_areas=num_largest_areas, fill=fill_holes, img_size=img_size
    )

    return resized_mask_femur, resized_mask_tibia


def extract_masks_from_png(file_path):
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
    color_to_label = {
        (0, 0, 0, 255): "background",  # Black
        (64, 64, 64, 255): "femur",  # Gray
        (192, 192, 192, 255): "tibia",  # White
    }

    # Step 4: Create binary masks for each label
    masks = []
    for color, label in color_to_label.items():
        # Create a binary mask for the current color
        mask = np.all(img_array == np.array(color), axis=-1).astype(np.uint8)
        masks.append(mask)

    return masks

def get_img_and_masks_right(file_path):
    return Image.open(file_path), extract_masks_from_png(file_path)


def print_image_matrix(file_path):
    """
    Loads a PNG image and prints its pixel values as a matrix.
    """
    # Step 1: Load the image
    img = Image.open(file_path)
    img_array = np.array(img)  # Convert to a NumPy array

    # Find unique RGBA values
    unique_colors = np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)
    print("Unique RGBA values in the image:", unique_colors)

    return img_array

# Example usage
if __name__ == "__main__":
    # Dummy inputs for demonstration
    file_path = "sag.png"
    # print_image_matrix(file_path)

    img, masks = get_img_and_masks_right(file_path)
    femur_mask, tibia_mask = process_right_leg(img, masks)
    np.savetxt("right_tibia_mask.txt", tibia_mask, fmt='%d')
    np.savetxt("right_femur_mask.txt", femur_mask, fmt='%d')
    print("Femur and Tibia masks processed.")
