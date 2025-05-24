import re
import ast

def get_image_name(initial_string):
   image_name = ""
   positions = [initial_string.find('_left'), initial_string.find('_right'), initial_string.find('_results'), initial_string.find('.')]
   positions_cln = [pos for pos in positions if pos != -1]
   position = min(positions_cln) if positions_cln else -1
   image_name = initial_string[:position]
   
   if(positions[0] != -1):
      which_side = 'left'
   elif(positions[1] != -1):
      which_side = 'right'
   else:
      which_side = None
   return image_name, which_side

def read_YOLO_data(file_path):
    data = {}
    
    with open(file_path, 'r') as file:
        # Read the header line to ignore it
        header = file.readline().strip()
        
        # Define a regex pattern to capture image name, correction angle, and points
        pattern = r'([^,]+),\s*([\d\.]+),\s*(\([^\)]+\)),\s*(\([^\)]+\)),\s*(\([^\)]+\)),\s*(\([^\)]+\)),\s*(\([^\)]+\)),\s*(\([^\)]+\))'
        
        # Iterate over the rest of the lines in the file
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            # Apply the regex pattern to extract the data
            match = re.match(pattern, line)
            if match:
                # Extract the matched groups
                image_name = match.group(1)
                image_name, side = get_image_name(match.group(1))
                image_name = image_name + '_' + side

                # Parse the points from the matched strings
               #  correction_angle = float(match.group(2))
                femur_head = ast.literal_eval(match.group(3))
               #  ost_point = ast.literal_eval(match.group(4))
                knee_inner = ast.literal_eval(match.group(5))
                knee_outer = ast.literal_eval(match.group(6))
                ankle_inner = ast.literal_eval(match.group(7))
                ankle_outer = ast.literal_eval(match.group(8))
                
                # Store the data in the dictionary
                data[image_name] = {
                  #   'Correction Angle': correction_angle,
                    'Femur Head': femur_head,
                  #   'OST Point': ost_point,
                    'Knee Inner': knee_inner,
                    'Knee Outer': knee_outer,
                    'Ankle Inner': ankle_inner,
                    'Ankle Outer': ankle_outer
                }
    
    return data

import os

def read_SEG_data(folder_path, data={}):
   
   # Iterate over all .txt files in the folder
   for file_name in os.listdir(folder_path):
      if file_name.endswith('.txt'):
         file_path = os.path.join(folder_path, file_name)
         
         # Extract the image name from the file name
         image_name, which_side = get_image_name(file_name)

         # Read the file content
         with open(file_path, 'r') as file:
            content = file.readlines()
         
         # Parse the points from the file content
         points = {}
         for line in content:
            line = line.strip()
            if not line:
               continue
            
            # Split the line into key and value
            key, value = line.split(':')
            key = key.strip()
            value = ast.literal_eval(value.strip())
            if key.find('Notch') != -1:
               points['Notch'] = (value[1], value[0])
            elif key.find('P1') != -1:
               points['P1'] = (value[1], value[0])
            elif key.find('P2') != -1:
               points['P2'] = (value[1], value[0])
         
            # Store the parsed data in the dictionary
            # DIKKAT: mask_to_notch algoritmasinda right-left karistigi icin 
            # burasi olmasi gerekenin tersi olacak 
            if key.find('Right') != -1:
               if image_name + '_left' in data:
                  data[image_name + '_left'].update(points)
               else:
                  data[image_name + '_left'] = points
            else:
               if image_name + '_right' in data:
                  data[image_name + '_right'].update(points)
               else:
                  data[image_name + '_right'] = points

   return data

def change_labels(is_left, data_dict):
   mapped_data = {}
    
   # Define the mapping of keys to labels
   key_mapping_shared = {
      'Femur Head': 'csp',
      # 'OST Point': 'labelOSTPoint',
      'Notch': 'cmtc',
      # could be worng, check the required labels
      'Knee Inner': 'cmb3', ##?
      'Knee Outer': 'cmb1', ##?
      'Ankle Inner': 'cb3', ##?
      'Ankle Outer': 'cb1', ##?
   }
   key_mapping_left = {
      'P1': 'cmt2', ###
      'P2': 'cmt1', ###
   }
   key_mapping_right = {
      'P1': 'cmt1', ###
      'P2': 'cmt2', ###
   }  

   for key, value in data_dict.items():
      if key in key_mapping_shared.keys():
         mapped_key = key_mapping_shared[key]
         mapped_data[mapped_key] = value
      elif key in key_mapping_left.keys() and is_left:
         mapped_key = key_mapping_left[key]
         mapped_data[mapped_key] = value
      elif key in key_mapping_right.keys() and not is_left:
         mapped_key = key_mapping_right[key]
         mapped_data[mapped_key] = value
   
   return mapped_data

from PIL import Image
TARGET_HEIGHT = 1025
def tranform_points(image_path, data):
  #  image_path = get_image_name(image_path)[0] + '.png'
  #  image_path = os.path.join(input_image_path, image_path)
   with Image.open(image_path) as img:
      width, height = img.size
      scale = TARGET_HEIGHT / float(height)
   for key, value in data.items():
      if isinstance(value, tuple) and len(value) == 2:
         data[key] = (round(float(value[0]) * scale, 3), round(float(value[1]) * scale, 3))


### LASTLY, RECORD AS JSON

# Example:
# data = {'P1':"asdsa", 'P2':"asdasd", 'Femur Head':"asdasd", 'OST Point':"asdasd", 'Knee Inner':"asdasd", 'Knee Outer':"asdasd", 'Ankle Inner':"asdasd", 'Ankle Outer':"asdasd"}
# data = map_keys_to_labels(is_left=True, data_dict=data)
# print(data)

import json
import shutil

# # Your new input dictionary
# points_dict = {
#     # Example data (use your real dictionary here)
#     "55457_grayscale_left": {
#         "csp": (260.309, 224.621),
#         "cmb3": (223.082, 610.186),
#         "cmb1": (286.759, 613.684),
#         # ...
#     },
#     "55457_grayscale_right": {
#         "csp": (120.078, 199.85),
#         "cmb3": (171.16, 593.112),
#         # ...
#     },
#     # More entries...
# }

def store_data(data):
   OUTPUT_JSON_FOLDER = "OUTPUT//JSON_RESULTS"
   # XRAY_JSON_FOLDER = r"C:\Users\saltu\Downloads\Compressed\7427_FILES\JsonVariables"
   os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)

   for full_key, points in data.items():
      filename_base = "_".join(full_key.split("_")[:1])  # e.g., "55457"
      side = "left" if full_key.endswith("left") else "right"

      # Prepare drawingProps
      drawingProps = {
         label: {"x": float(x), "y": float(y)}
         for label, (x, y) in points.items()
      }

      # JSON content
      json_data = {
         "drawingProps": drawingProps,
         "type": "Pointizr.Objects.ExtremityMechanic",
         "name": "Orthorontgenogram",
         "erasable": True,
         "selectable": True,
         "locked": False,
         "measurementValues": [],
         "objectModelVersion": 0.2,
      }

      # Create folder and paths
      sub_folder = os.path.join(OUTPUT_JSON_FOLDER, filename_base)
      os.makedirs(sub_folder, exist_ok=True)

      json_path = os.path.join(sub_folder, f"{filename_base}_{side}.json")
      with open(json_path, "w") as f:
         json.dump(json_data, f, indent=2)

      # Copy xray_position.json if it exists
      #  xray_position_source = os.path.join(XRAY_JSON_FOLDER, filename_base, "xray_position.json")
      #  xray_position_dest = os.path.join(sub_folder, "xray_position.json")
      #  if os.path.exists(xray_position_source):
      #      shutil.copy2(xray_position_source, xray_position_dest)
