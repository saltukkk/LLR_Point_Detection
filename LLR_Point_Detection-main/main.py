import argparse
import time

parser = argparse.ArgumentParser(description="Run YOLO + Segmentation Pipeline")
parser.add_argument('--mode', type=str, choices=['all', 'segmentation'], default='all',
               help="Mode to run the pipeline: 'all' or 'segmentation' only")
args = parser.parse_args()

# Dictionary to store timestamps
timestamps = {}

import subprocess

yolo_cmds = r"""
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate yolo_env
python yolo_whole.py
"""

seg_cmds_1 = r"""
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate segmentation_env
python main.py --id 003 --clear
"""

seg_cmds_2 = r"""
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate segmentation_env
python calculate_for_whole_folder2.py
"""

import os
import sys
import shutil

def copy_images(src_dir, dest_dir):
   """
   Copy images from src_dir to dest_dir.
   """
   os.makedirs(dest_dir, exist_ok=True)
   print(dest_dir)

   # Check if input images directory exists
   if os.path.exists(src_dir):
     for file_name in os.listdir(src_dir):
       file_path = os.path.join(src_dir, file_name)
       if os.path.isfile(file_path):
            print(file_path)
            shutil.copy(file_path, dest_dir)
   else:
     print(f"Directory {src_dir} does not exist.")

def clean_up():

   # Clean up: remove the transit directory after processing
   dirs_to_clean = [
      os.path.join(os.getcwd(), "./SEGMENTATION/bone_seg_nnunet_main/testing_images"),
      os.path.join(os.getcwd(), "./SEGMENTATION/bone_seg_nnunet_main/postprocessed"),
      os.path.join(os.getcwd(), "./SEGMENTATION/bone_seg_nnunet_main/predictions"),
      os.path.join(os.getcwd(), "./SEGMENTATION/bone_seg_nnunet_main/jsons"),
      os.path.join(os.getcwd(), "./OUTPUT/"),
   ]


   for d in dirs_to_clean:
      if os.path.exists(d):
         if os.path.isfile(d):
               os.remove(d)
         else:
               shutil.rmtree(d)
         print(f"Cleaned the object at start: {d}")

# Clean up: remove the transit directory at the start
clean_up()

if args.mode == "all":
   # 1- YOLO
   script_dir = os.path.join(os.getcwd(), "./YOLO/demo")
   print(f"Running YOLO in {script_dir}")

   process = subprocess.Popen(yolo_cmds, shell=True, executable="/bin/bash", cwd=script_dir)
   while True:
     exit_code = process.poll()
     if exit_code is not None:
       print(f"YOLO terminated with exit code: {exit_code}")
       timestamps["YOLO_end"] = time.time()
       break
     else:
       print("YOLO still running...")
       time.sleep(10)

if args.mode in ["all", "segmentation"]:
   # 2- SEGMENTATION steps

   # COPY TEST_IMAGES
   print("Copying INPUT_IMAGES into the SEGMENTATION folder...")
   segmentation_dir = os.path.join(os.getcwd(),"./SEGMENTATION/bone_seg_nnunet_main")
   input_images_dir = "./INPUT_IMAGES"
   testing_images_dir = os.path.join(segmentation_dir, "./testing_images")

   if os.path.exists(testing_images_dir):
     if not os.path.isdir(testing_images_dir):
       os.remove(testing_images_dir)
     else:
       shutil.rmtree(testing_images_dir)
   copy_images(input_images_dir, testing_images_dir)

   # Run first segmentation process
   script_dir = segmentation_dir
   print(f"Running segmentation in {script_dir}")
   process = subprocess.Popen(seg_cmds_1, shell=True, executable="/bin/bash", cwd=script_dir)
   while True:
     exit_code = process.poll()
     if exit_code is not None:
       print(f"SEGMENTATION terminated with exit code: {exit_code}")
       timestamps["SEGMENTATION_phase1_end"] = time.time()
       break
     else:
       print("SEGMENTATION still running...")
       time.sleep(10)

   # Run second segmentation command (json generation)
   cmd_seg_json = r"""
   CONDA_BASE=$(conda info --base)
   if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
     source "$CONDA_BASE/etc/profile.d/conda.sh"
   else
     echo "ERROR: conda.sh not found" >&2; exit 1
   fi
   conda activate segmentation_env
   python main_json.py --clear
   """
   print(f"Running segmentation json command in {segmentation_dir}")
   process = subprocess.Popen(
     cmd_seg_json,
     shell=True,
     executable="/bin/bash",
     cwd=segmentation_dir
   )
   process.communicate()
   while True:
     exit_code = process.poll()
     if exit_code is not None:
       # Copy jsons to OUTPUT/JSON_SEGMENTATION
       os.makedirs(os.path.join("./", "./OUTPUT"), exist_ok=True)
       os.makedirs(os.path.join("./", "./OUTPUT/JSON_SEGMENTATION"), exist_ok=True)
       copy_images(os.path.join(segmentation_dir, "jsons"), os.path.join("./", "./OUTPUT/JSON_SEGMENTATION"))
       print(f"MASK_TO_JSON terminated with exit code: {exit_code}")
       timestamps["SEGMENTATION_phase2_end"] = time.time()
       break
     else:
       print("MASK_TO_JSON still running...")
       time.sleep(10)

if args.mode == "all":
   # 3- Point detection segmentation step
   script_dir = "./SEGMENTATION/mask_to_notch"
   print(f"Running point detection in {script_dir}")
   process = subprocess.Popen(seg_cmds_2, shell=True, executable="/bin/bash", cwd=script_dir)
   while True:
     exit_code = process.poll()
     if exit_code is not None:
       print(f"SEGMENTATION-PHASE2[Point Detection] terminated with exit code: {exit_code}")
       timestamps["POINT_DETECTION_end"] = time.time()
       break
     else:
       print("SEGMENTATION-PHASE2[Point Detection] still running...")
       time.sleep(10)

   from read_results import read_YOLO_data, read_SEG_data, tranform_points, change_labels, store_data, get_image_name
   import os

   yolo_file_path = './OUTPUT//OUTPUT_YOLO//results.txt'
   seg_folder = './OUTPUT//OUTPUT_SEGMENTATION//results'
   input_images_path = './INPUT_IMAGES'

   print("OUTPUT post-processing to json format...")
   data = read_YOLO_data(yolo_file_path)
   data.update(read_SEG_data(seg_folder,data))

   mapped_data = {}
   for key, value in data.items():
     is_left = False
     if key.find('left') != -1:
       data[key]['is_left'] = True
     else:
       data[key]['is_left'] = False
     img_path = os.path.join(input_images_path, get_image_name(key)[0]+'.png')
     if os.path.exists(img_path):
       tranform_points(image_path=img_path, data=value)
       mapped_data[key] = change_labels(is_left=data[key]['is_left'], data_dict=value)

   store_data(mapped_data)

# Finally, print out all recorded timestamps in a human-readable format.
print("\nTime Marks:")
for stage, ts in timestamps.items():
   # Format the timestamp
   formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
   print(f"{stage}: {formatted_time}")

