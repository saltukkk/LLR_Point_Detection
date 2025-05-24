import subprocess

yolo_cmds_1 = "conda create -n yolo_env python=3.11 -y"

yolo_cmds_2 = r"""
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate yolo_env
conda install scipy==1.10.0 -y
conda install -c conda-forge menpo -y
pip install menpofit --disable-pip-version-check --no-python-version-warning --root-user-action=ignore -q --no-color
conda install tensorflow==2.12 -y
conda install scikit-image -y
conda install scikit-learn -y
pip install --root-user-action=ignore tensorflow-addons==0.20.0 --disable-pip-version-check --no-python-version-warning --root-user-action=ignore -q --no-color
pip install pillow --disable-pip-version-check --no-python-version-warning --root-user-action=ignore -q --no-color
"""

seg_cmds_1 = "conda create -n segmentation_env python=3.11 -y"

seg_cmds_2 = r"""
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate segmentation_env
conda install matplotlib -y
conda install pytorch==2.3 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy=1.24 -y
pip install opencv-python-headless --disable-pip-version-check --no-python-version-warning --root-user-action=ignore -q --no-color
pip install pillow --disable-pip-version-check --no-python-version-warning --root-user-action=ignore -q --no-color
"""

extra = "pip install pillow --disable-pip-version-check --no-python-version-warning --root-user-action=ignore -q --no-color"

process = subprocess.Popen(yolo_cmds_1, shell=True)

# Wait for output file
import time
while True:
   exit_code = process.poll()
   if exit_code is not None:
      print(f"YOLO conda env constructed with exit code: {exit_code}")
      break
   else:
      print("YOLO conda env building...")
      time.sleep(10)
import os
import sys

# 1- YOLO
script_dir = os.path.join(os.getcwd(), "./YOLO/demo")
process = subprocess.Popen(yolo_cmds_2, shell=True, executable="/bin/bash", cwd=script_dir)

# Wait for output file
import time
while True:
   exit_code = process.poll()
   if exit_code is not None:
      print(f"YOLO env dependencies built with exit code: {exit_code}")
      break
   else:
      print("YOLO env dependencies building...")
      time.sleep(10)

# 2- Segmentation
segmentation_dir = os.path.join(os.getcwd(),"./SEGMENTATION/bone_seg_nnunet_main")
script_dir = segmentation_dir
process = subprocess.Popen(seg_cmds_1, shell=True)

while True:
   exit_code = process.poll()
   if exit_code is not None:
      print(f"SEGMENTATION conda env constructed with exit code: {exit_code}")
      break
   else:
      print("SEGMENTATION conda env building...")
      time.sleep(10)


process = subprocess.Popen(seg_cmds_2, shell=True, executable="/bin/bash", cwd=script_dir)
while True:
   exit_code = process.poll()
   if exit_code is not None:
      print(f"SEGMENTATION env dependencies built with exit code: {exit_code}")
      break
   else:
      print("SEGMENTATION env dependencies building...")
      time.sleep(10)


process = subprocess.Popen(extra, shell=True)

# Wait for output file
import time
while True:
   exit_code = process.poll()
   if exit_code is not None:
      print(f"Pillow is installed with exit code: {exit_code}")
      break
   else:
      print("Pillow is installing...")
      time.sleep(10)
