# Bone Segmentation

This code is based on the nnUNet Neural Network structure.
A x-ray image is taken as an input and a segmentation mask is produced. 
The extraction of the segmentation is done using a pre-trained model. Runtime may vary depending on the image.
Model should be downloaded from drive link and unzipped into the repository folder.
[https://drive.google.com/file/d/1qxn6DxiM09tPoGZhj-ifajVVit9Vo9WZ/view?usp=sharing](https://drive.google.com/file/d/1VD2Ot9Y2-kkBoUVAIwUd-z1ahKN5qsNr/view?usp=drive_link)

## Setup

1. Create and activate a new conda environment using python 3.10
```
conda create -n new_env python=3.10
conda activate new_env
```

2. Install matplotlib, pytorch, and opencv. Chech your cuda version for pytorch
```
conda install matplotlib
conda install pytorch==2.1.2 torchvision=0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
```

3. Unzip model from drive link above to the repository folder

## Inputs and Outputs

Image: any **.png file**

Output: 
Mask: .png file with same size as the input.  (Output is in postporcessed folder)

Json file: .json file for all ploygons generated for the image (Output file name is same as input file)

## Usage

For generating predictions, put all images to testing_images folder and run as follows:
```python
python main.py --id model-id
```
Parameters:
* --id: model id to use for prediction. Use 3 number format (default 006)
* --clear: Clear the output directory before predicting new images (optional)
* --new: Use to install nnunet again in new conda environments (if not done automatically) (optional)

For generating Json File (Assumes Mask is already present)
```python
python main_json.py --clear
```
Parameters:
* --clear: Clear the jsons folder before adding new json files

For training new model given the images in training_images and jsons in training_jsons and/or labels in training_labels
```python
python train.py --id new-model-id --epochs number-of-epochs
``` 
Parameters:
* --id: id of a new model in 3 number format (default 004)

## Requirements
Included in the file "requirements.txt".
Additionally, git should be installed.

* --epochs: the number of epochs for training (default 100)
* --clear: Clear the old train dataset (optional)

Extra explain;
main.py: testing_images klasöründeki fotoları predict edip postprocessed klasörüne yazıyor.
main_json.py: postprocessed klasöründeki predictionları alıp jsona dönüştürerek jsons klasörüne yazıyor.
train.py: train_images`daki fotoları ve, ya train_labels`daki labelleri ya da train_jsons`daki jsonları labela dönüştürerek model eğitiyor.


