from argparse import ArgumentParser
import os
import shutil
from distutils.dir_util import copy_tree
import glob
import json
import numpy as np
from PIL import Image, ImageDraw
import re

def json_to_mask():
	# Define class labels and their corresponding colors (RGB format)
	class_colors = {
		'femur-left': 1,
		'femur-right': 2,
		'tibia-left': 3,
		'tibia-right': 4
	}
	os.makedirs('training_labels', exist_ok=True)
	for json_file_path in (glob.glob("training_jsons/*.json")):
		with open(json_file_path, 'r') as json_file:
			data = json.load(json_file)

		# Define image dimensions (height and width)
		image_height = data["imageHeight"]
		image_width = data["imageWidth"]

		# Initialize a blank composite mask
		composite_mask = np.zeros((image_height, image_width), dtype=np.uint8)

		# Populate the composite mask with class colors
		for shape in data["shapes"]:
			if (shape["label"].split('-').__len__() < 2):
				continue
			class_label = shape["label"].split('-')[0]+'-'+shape["label"].split('-')[1]
			class_color = class_colors.get(class_label)
			if class_color is None:
				continue

			segment_points = shape["points"]
			for i in range(len(segment_points)):
				segment_points[i][0] = int(segment_points[i][0])
				segment_points[i][1] = int(segment_points[i][1])
				segment_points[i] = tuple(segment_points[i])
			mask_image = Image.fromarray(composite_mask)
			draw = ImageDraw.Draw(mask_image)
			draw.polygon(segment_points, outline=class_color, fill=class_color)
			composite_mask = np.array(mask_image, dtype=np.uint8)
		composite_mask_image = Image.fromarray(composite_mask)
		composite_mask_image.save('training_labels/'+json_file_path[15:-5]+'.png')


# def convert_to_grayscale(directory):
# 	for picture in os.listdir(directory):
# 		image = os.path.join(directory, picture)
# 		if picture == "desktop.ini":
# 			os.remove(image)

# 		img = Image.open(image).convert('L')
# 		img.save(image.split('.')[0]+"_0000.png")
# 		os.remove(image)

def transfer_and_grayscale(directory, save_path):
	for picture in os.listdir(directory):
		image = os.path.join(directory, picture)
		if picture == "desktop.ini":
			os.remove(image)

		img = Image.open(image).convert('L')
		img.save(f"{save_path}/{picture.split('.')[0]}_0000.png")


def modify_file(file_path, num):
    # Read the content of a.py
    with open(file_path, 'r') as file:
        lines = file.readlines()

    pattern = r'(\s*self.num_epochs\s*=\s*)\d+'

    # Modify the specific line
    for i in range(len(lines)):
        if re.search(pattern, lines[i]):
            lines[i] = re.sub(pattern, r'\g<1>'+str(num), lines[i])
            break

    # Write the modified content back to a.py
    with open(file_path, 'w') as file:
        file.writelines(lines)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--id', metavar='int', nargs=1, default=['004'], help='ID of the model in three number format (e.g. 003)')
	parser.add_argument('--clear', action="store_true", help='Clear the old train image and labels')
	parser.add_argument('--epochs', metavar='int', nargs=1, default=[100], help='Number of epochs to train the model. Default is 100.')
	args = parser.parse_args()
	
	dataset_name = 'Dataset'+ args.id[0] +'_FemurTibia'
	base_dir = os.getcwd()
	
	if not os.path.isdir("nnUNet"):
		os.system('git clone https://github.com/MIC-DKFZ/nnUNet.git')
		respository_dir = os.path.join(base_dir,'nnUNet')
		os.chdir(respository_dir)
		os.system('pip install -e .')
		os.makedirs('nnunetv2/nnUNet_trained_models', exist_ok=True)
		shutil.move(os.path.join(base_dir,'Dataset003_FemurTibia'), 'nnunetv2/nnUNet_trained_models')

	os.chdir(base_dir)
	
	main_dir = os.path.join(base_dir,'nnUNet/nnunetv2')
	raw_data_base_path = os.path.join(main_dir,'nnUNet_raw_data_base')
	preprocessed_path = os.path.join(main_dir,'preprocessed')
	result_folder_path = os.path.join(main_dir,'nnUNet_trained_models')
	os.makedirs(raw_data_base_path, exist_ok=True)
	os.makedirs(preprocessed_path, exist_ok=True)
	os.makedirs(result_folder_path, exist_ok=True)
	os.environ['nnUNet_raw'] = raw_data_base_path
	os.environ['nnUNet_preprocessed'] = preprocessed_path
	os.environ['nnUNet_results'] = result_folder_path
	dataset_path = os.path.join(raw_data_base_path, dataset_name)
	train_images_path = os.path.join(raw_data_base_path, dataset_name, "imagesTr")
	train_labels_path = os.path.join(raw_data_base_path, dataset_name, "labelsTr")
	
	print("Converting json files to mask images...")
	json_to_mask()
	print("Json files converted to mask images")

	os.makedirs(dataset_path, exist_ok=True)
	os.makedirs(train_images_path, exist_ok=True)
	os.makedirs(train_labels_path, exist_ok=True)
	
	print("Transfering images and labels to the dataset folder...")
	if args.clear:
		for filename in os.listdir(train_images_path):
			os.remove(os.path.join(train_images_path, filename))
		for filename in os.listdir(train_labels_path):
			os.remove(os.path.join(train_labels_path, filename))

	transfer_and_grayscale('training_images', train_images_path)
	copy_tree('training_labels', train_labels_path)
	print("Images and labels transfered to the dataset folder")

	# convert_to_grayscale(train_images_path)

	from  nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
	generate_dataset_json(dataset_path, channel_names = {0 : "X-Ray"},
						labels = {
							"background": 0,
							"Femur Left": 1, #Femur Left
							"Femur Right": 2, #Femur Right
							"Tibia Left": 3, #Tibia Left
							"Tibia Right": 4, #Tibia Right
						},
						num_training_cases=len(os.listdir(train_images_path)),
						file_ending=".png",
						dataset_name=dataset_name)

	os.system(f"nnUNetv2_plan_and_preprocess -d {args.id[0]} --verify_dataset_integrity --clean")

	modify_file(r'nnUNet\nnunetv2\training\nnUNetTrainer\nnUNetTrainer.py', args.epochs[0])

	os.system(f"nnUNetv2_train {args.id[0]} 2d 0 --npz -tr nnUNetTrainerNoMirroring --c")
	os.system(f"nnUNetv2_train {args.id[0]} 2d 1 --npz -tr nnUNetTrainerNoMirroring --c")
	os.system(f"nnUNetv2_train {args.id[0]} 2d 2 --npz -tr nnUNetTrainerNoMirroring --c")
	os.system(f"nnUNetv2_train {args.id[0]} 2d 3 --npz -tr nnUNetTrainerNoMirroring --c")
	os.system(f"nnUNetv2_train {args.id[0]} 2d 4 --npz -tr nnUNetTrainerNoMirroring --c")
	os.system(f"nnUNetv2_find_best_configuration {args.id[0]} -c 2d -tr nnUNetTrainerNoMirroring")