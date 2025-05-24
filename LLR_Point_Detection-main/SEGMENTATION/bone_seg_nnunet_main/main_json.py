import json
import numpy as np
import os
import cv2
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--clear', action="store_true", help='clear the jsons folder before adding new json files')
	args = parser.parse_args()

	# Create the "jsons" folder if it doesn't exist
	os.makedirs("jsons", exist_ok=True)

	postprocessing_path = "postprocessed"
	for filename in os.listdir(postprocessing_path):
		mask = os.path.join(postprocessing_path, filename)
		mask = cv2.imread(mask, 0)
		mask1 = np.where(mask == 64, 64, 0).astype(np.uint8)
		mask2 = np.where(mask == 128, 128, 0).astype(np.uint8)
		mask3 = np.where(mask == 192, 192, 0).astype(np.uint8)
		mask4 = np.where(mask == 255, 255, 0).astype(np.uint8)

		contour1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour4, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		shapes = []

		for contour in contour1:
			if contour.shape[0] < 3:
				continue
			points = contour.squeeze().tolist()
			shape = {
				"label": "femur left",
				"points": points,
				"shape_type": "polygon"
			}
			shapes.append(shape)

		for contour in contour2:
			if contour.shape[0] < 3:
				continue
			points = contour.squeeze().tolist()
			shape = {
				"label": "femur right",
				"points": points,
				"shape_type": "polygon"
			}
			shapes.append(shape)

		for contour in contour3:
			if contour.shape[0] < 3:
				continue
			points = contour.squeeze().tolist()
			shape = {
				"label": "tibia left",
				"points": points,
				"shape_type": "polygon"
			}
			shapes.append(shape)
		
		for contour in contour4:
			if contour.shape[0] < 3:
				continue
			points = contour.squeeze().tolist()
			shape = {
				"label": "tibia right",
				"points": points,
				"shape_type": "polygon"
			}
			shapes.append(shape)

		json_data = {
			"imageHeight": mask.shape[0],
			"imageWidth": mask.shape[1],
			"shapes": shapes
		}
		# Save the JSON data to a file inside the "jsons" folder
		json_file = os.path.join("jsons", filename[:-4] + ".json")
		with open(json_file, 'w') as f:
			json.dump(json_data, f, indent=4)
    