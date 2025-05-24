from argparse import ArgumentParser
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

def transfer_and_grayscale(directory, save_path):
	for picture in os.listdir(directory):
		image = os.path.join(directory, picture)
		if picture == "desktop.ini":
			os.remove(image)

		img = Image.open(image).convert('L')
		img.save(f"{save_path}/{picture.split('.')[0]}_0000.png")


if __name__ == "__main__":
	parser = ArgumentParser()
	# parser.add_argument('--image_name', metavar='str', nargs=1, default=['sample.jpg'], help='Name of the image')
	parser.add_argument('--id', metavar='str', nargs=1, default=['003'], help='ID of the model in three number format (e.g. 003)')
	parser.add_argument('--clear', action="store_true", help='Clear the output directory before predicting new images')
	parser.add_argument('--new', action="store_true", help='Redownload the nnUNet repository if in new environment')
	args = parser.parse_args()

	

	dataset_name = 'Dataset'+ args.id[0] +'_FemurTibia'

	base_dir = os.getcwd()

	if not os.path.isdir("nnUNet") or args.new:
		os.system('git clone https://github.com/MIC-DKFZ/nnUNet.git')
		respository_dir = os.path.join(base_dir,'nnUNet')
		os.chdir(respository_dir)
		os.system('pip install -e .')
		os.makedirs('nnunetv2/nnUNet_trained_models', exist_ok=True)
	
	os.chdir(base_dir)
	
	if os.path.isdir(dataset_name):
		shutil.move(os.path.join(base_dir, dataset_name), 'nnUNet/nnunetv2/nnUNet_trained_models')
		
	

	os.makedirs(f'nnUNet/nnunetv2/nnUNet_raw_data_base/{dataset_name}/imagesTs', exist_ok=True)
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
	test_images_path = os.path.join(raw_data_base_path, dataset_name, "imagesTs")
	
	predictions_path = "predictions"
	postprocessing_path = "postprocessed"
	os.makedirs(predictions_path, exist_ok=True)
	os.makedirs(postprocessing_path, exist_ok=True)
	
	# for filename in os.listdir(predictions_path):
		# if filename == "dataset.json" or filename == "plans.json" or filename == "predict_from_raw_data_args.json":
		#	 continue
		# os.remove(os.path.join(predictions_path, filename))
	
	print("Clearing directories...")
	for filename in os.listdir(test_images_path):
		os.remove(os.path.join(test_images_path, filename))
	for filename in os.listdir(predictions_path):
		os.remove(os.path.join(predictions_path, filename))

	if args.clear:
		for filename in os.listdir(postprocessing_path):
			os.remove(os.path.join(postprocessing_path, filename))

	
	
	print("Transfering and grayscaling images...")
	transfer_and_grayscale('testing_images', test_images_path)

	# copy_tree('testing_images', test_images_path)
	# for image in os.listdir(test_images_path):
	# 	if image == "desktop.ini":
	# 		os.remove(os.path.join(test_images_path, image))
	# 	else:
	# 		img = Image.open(os.path.join(test_images_path, image))
	# 		img.save(os.path.join(test_images_path, image.split('.')[0]+"_0000.png"))
	# 		os.remove(os.path.join(test_images_path, image))
	
	os.environ["nnUNet_def_n_proc"] = "12"

	if torch.cuda.is_available():
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = ""


	os.system(f'nnUNetv2_predict -i nnUNet/nnunetv2/nnUNet_raw_data_base/{dataset_name}/imagesTs -o predictions -d {args.id[0]} -c 2d -tr nnUNetTrainerNoMirroring --disable_tta')
	
	# from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
	# from nnUNet.nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
	# if torch.cuda.is_available():
	# 	predictor = nnUNetPredictor(use_mirroring=False, device = torch.device('cuda'))
	# else:
	# 	predictor = nnUNetPredictor(use_mirroring=False, device = torch.device('cpu'))
	
	# predictor.initialize_from_trained_model_folder(model_training_output_dir = f"nnUNet/nnunetv2/nnUNet_trained_models/{dataset_name}/nnUNetTrainerNoMirroring__nnUNetPlans__2d",
	# 										 use_folds = (0, 1, 2, 3, 4),
	# 										 checkpoint_name = 'checkpoint_final.pth')
	# img, props = NaturalImage2DIO().read_images([os.path.join(test_images_path, args.image_name[0][:-4]+"_0000.png")])
	# ret = predictor.predict_single_npy_array(img, props, None, None, False)
	# plt.imsave(os.path.join(predictions_path, args.image_name[0]), ret[0], vmin=0, vmax=4, cmap='gray')
	os.system(f'nnUNetv2_apply_postprocessing -i predictions -o postprocessed -pp_pkl_file nnUNet/nnunetv2/nnUNet_trained_models/{dataset_name}/nnUNetTrainerNoMirroring__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json nnUNet/nnunetv2/nnUNet_trained_models/{dataset_name}/nnUNetTrainerNoMirroring__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json')

	# turn images from 0-4 grayscale to 0-255 grayscale using vmin vmax
	for filename in os.listdir(postprocessing_path):
		image = os.path.join(postprocessing_path, filename)
		img = Image.open(image)
		if np.asarray(img).max() > 4:
			continue
		plt.imsave(image, img, vmin=0, vmax=4, cmap='gray')
