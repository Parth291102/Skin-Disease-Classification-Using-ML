# Advanced Techniques for Skin Disease Classification Using Machine Learning
![Python Version](https://img.shields.io/badge/Python-3.11.5-blue)
![GPU](https://img.shields.io/badge/GPU-Google%20T4-brightgreen)
![Gradio Version](https://img.shields.io/badge/Gradio-v5.6.0-purple)

## Contents
- [Dataset Description](#Dataset-Description)
- [Tech Stack](#Technologies-Used)
- [Installation](#Setup-Instructions) 
- [Demo](#Demo)
- [Contributors](#Contributors)


## Dataset Description
We are using the Mpox Skin Lesion Dataset (version 2.0) from Kaggle, which contains images of skin lesions potentially caused by Mpox.  MSLD v2.0 features a collection of images categorized into six classes: Mpox (284 images), Chickenpox (75 images), Measles (55 images), Cowpox (66 images), Hand-foot-mouth disease (HFMD) (161 images), and Healthy (114 images). This dataset comprises a total of 755 original skin lesion images collected from 541 unique patients, providing a well-rounded sample. Notably, professional dermatologists have endorsed this latest version, which has also received approval from relevant regulatory authorities.
However, due to the dataset’s relatively small size, we plan to extend it by integrating images from other relevant datasets or by performing data augmentation. This combined dataset will undergo various data augmentation and GANN techniques, such as rotation, translation, reflection, shear, hue, saturation, contrast, brightness jitter, noise, and scaling to ensure sufficient data for model training and to address potential issues like overfitting.

Dataset Link: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20?select=Original+Images

## Technologies Used
1. Python
2. Google Colab - T4 GPU: System RAM --> 12.7 GB, GPU RAM --> 15 GB, Disk --> 112.6 GB

## Setup Instructions
1. Clone the repository:
 ```
  git clone https://github.ncsu.edu/pparikh2/ALDA_Project_Group21
```  
2. Start a terminal. Run the following command to install the required dependencies:
```
  pip install -r requirements.txt
```
3. In the directory where this repo has been cloned, please run the below command to execute a bash script to run the Gradio Interface:
```
   python3 Project_Interface.py
```

## Demo

https://github.ncsu.edu/pparikh2/ALDA_Project_Group21/assets/35501/09b5b9a0-690c-4470-8da0-f204b9a7f070

## Contributors
1. Parth Parikh - pparikh2 - pparikh2@ncsu.edu
2. Mrudani Hada - mhada2 - mhada2@ncsu.edu
3. Abhinav Jami - ajami3 - ajami3@ncsu.edu
4. Yuvraj Bhatia - ybhatia2 - ybhatia2@ncsu.edu




