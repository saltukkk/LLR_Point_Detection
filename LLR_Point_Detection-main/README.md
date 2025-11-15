# LLR_Point_Detection
# 1. Requirements
GPU is required. Also Anaconda software is required. 

# 2. Download the Ready Models

Inside the repo, download the ready models:

1- Dataset003_FemurTibia.zip:

```curl -L -C - --progress-bar -o Dataset003_FemurTibia.zip "https://drive.usercontent.google.com/download?id=1y3bW4CPER2YmZYJxDZRhLeHTG9VPuzQ0&export=download&authuser=0&confirm=t&uuid=2b6ea6ec-31a6-476b-8197-3ea8ec90e2d2&at=APcmpoy2Z3YD79gc6QT50wxq0Tyu%3A1744993960061"```

2- model_28.07.2022_3_classes_p6.h5

```curl -L -C - --progress-bar -o model_28.07.2022_3_classes_p6.h5 "https://drive.usercontent.google.com/download?id=1pM9Zn1W-rYBYszmH1zmIq4Pkbcc5OXi5&export=download&authuser=0&confirm=t&uuid=4587dd22-da07-413b-a536-e6e2cb29b9a3&at=APcmpoxTS37waxp99KQZX_EX1Fd3%3A1744994244292"```

3- aam_full_h_59_right.p:

```curl -L -C - --progress-bar -o aam_full_h_59_right.p "https://drive.usercontent.google.com/download?id=1vb_-lNFdMcZzrNfPgsKHM4JffdsFku3_&export=download&authuser=0&confirm=t&uuid=f7198757-1d9c-4a6e-99cb-07f0bf804335&at=APcmpowtiqPvJ8NBGn_W2-2XoEof%3A1744994280290"```

4- aam_full_h_59_left.p:

```curl -L -C - --progress-bar -o aam_full_h_59_left.p "https://drive.usercontent.google.com/download?id=1akIr_lZlVpryw8R_7X96E_W0nq2NbjuG&export=download&authuser=0&confirm=t&uuid=abdfd9e6-9ad0-4fae-9b01-c1cd37e01561&at=APcmpoykDkhjUc40uD5pf0Fz1tPB%3A1744994309536"```

5-	Lastly, move models

```
unzip Dataset003_FemurTibia.zip
mv Dataset003_FemurTibia ./SEGMENTATION/bone_seg_nnunet_main
mv model_28.07.2022_3_classes_p6.h5 ./YOLO/models
mv aam_full_h_59_right.p ./YOLO/models
mv aam_full_h_59_left.p ./YOLO/models
```

# 3. Run setup.py
For one time only, run setup.py program and construct the environments.
```
python setup.py
```

# 4. Input Images
Put your input images into the ./INPUT_IMAGES folder, then run main.py

# 5. Run the main.py Program
Run all the pipeline
```
python main.py --mode all
```

Or run only the segmentation
```
python main.py --mode segmentation
```
# Outputs
Output images of YOLO model can be found at ./OUTPUT/YOLO

Output images of Segmentation model, can be found at ./OUTPUT/SEGMENTATION/test

Specific points in the Json format can be found at ./OUTPUT/JSON_RESULTS

All segmentation points can be found at ./OUTPUT/JSON_SEGMENTATION

# Theoretical Explaination and Presentation

![Poster_Submit](https://github.com/user-attachments/assets/fda911fd-62b7-4da6-81e3-c296d3dd32e7)

# Clinical Deployment to Server

https://github.com/user-attachments/assets/884d5176-1241-4908-8855-22da08523322

