# SPACE SIEVE

## Introduction
We aim to develop a model that identifies objects in space as active satellites or space debris by analyzing space imagery. This requires differentiating between operational equipment and non-operational pieces or debris that are in orbit.

By integrating two distinct datasets, our goal is to incorporate debris of various sizes to ensure our analysis is resilient to the challenges presented by collisions and the resulting breakup of debris.

It should be of concern to all. Whether it's someone using a smartphone, checking the weather, or navigating with GPS, satellites are integral to these services. However, the increasing clutter of defunct satellites and debris in space poses a significant risk of collision with operational satellites, disrupting these essential services. The addition of more satellites into an already congested space heightens these risks and complicates the expansion of global communication networks. Furthermore, this initiative is crucial in tackling the issue of space debris through strategies like Active Debris Removal (ADR) and the use of Solar Sails.

[Watch the video demonstration of our project](images%20and%20videos/video.mp4)

## Table of Contents
- [Installation & Setup](#installation--setup)
  - [Python Environment Setup](#python-environment-setup)
    - [Prerequisites](#prerequisites)
    - [Virtual Environment](#virtual-environment)
    - [Install Dependencies](#install-dependencies)
  - [Data](#data)
  - [Data Preprocessing](#data-preprocessing)
  - [Image Preprocessing](#image-preprocessing)
  - [Object Detection](#object-detection)
  - [CNN Model](#cnn-model)
  - [User Interface Setup](#user-interface-setup)
  - [Yolov5 Model Setup](#yolov5-model-setup)
    - [Data Folder Setup](#data-folder-setup)
    - [Configuration](#configuration)
    - [Training](#training)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Interface](#interface)
- [Contributors](#contributors)


## INSTALLATION & SETUP

## Python Environment Setup

### Prerequisites

Ensure you have the following software installed on your system:
- Python (version 3.x)
- pip (Python package installer)

### Virtual Environment

It is recommended to use a virtual environment to manage the dependencies for your project. Follow the steps below to set up a virtual environment.

1. **Install `virtualenv` package (if not already installed):**
    ```bash
    pip install virtualenv
    ```

2. **Create a virtual environment:**
    Navigate to your project directory and run:
    ```bash
    virtualenv Venv
    ```

3. **Activate the virtual environment:**

    - **Windows:**
      ```bash
      .\Venv\Scripts\activate
      ```

    - **macOS/Linux:**
      ```bash
      source Venv/bin/activate
      ```

### Install Dependencies

Once the virtual environment is activated, install the necessary dependencies using the `requirements.txt` file.

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## Data

In your project directory create a folder named **data** and load the datasets here

The datasets **SPARK2022** and **SPARK2021** are invaluable assets for the field of spacecraft detection and trajectory estimation, hosted by the University of Luxembourg's CVI² at SnT. 

- **URL:** [SPARK2022 Dataset](https://cvi2.uni.lu/spark-2022-dataset) and [SPARK2021 Dataset](https://cvi2.uni.lu/spark-2021-dataset)
- **Content:** Roughly 110,000 high-resolution RGB images.
- **Features:** Images annotated with object bounding boxes and class labels.

Unzip both files in the same locations.

Rename the files as `2021` and `2022` accordingly.
 

## Data Preprocessing

- Data balancing
- Image Sequencing to avoid duplicates
- Image renaming, relocating, resizing and decoding
- Final `master_data` setup

Open the file data_ingestion_v2.ipynb

> **_File Path Check_**: `data/2021` and `data/2022`

To run the notebook data_ingestion_v2.ipynb, execute the below command from the parent directory

```bash
jupyter notebook data_ingestion_v2.ipynb
```
Files for reference: `Data_Preprocessing_2021.ipynb`, `Data_Preprocessing_2022.ipynb`, `Data_Preprocessing_2022_functionized.ipynb` and `Data_Preprocessing.py`. 

- By now we should have a **master_data** folder inside **data** folder

## Image Preprocessing

- Image Resizing
- Image Denoising (Non-Local Means)
- Image Contrast Enhancement (CLAHE)
- Object Detection (Plotting Boundary Box) 
- Storing the Preprocessed images in the newly created folder

Open the notebook Image Preprocessing.ipynb

> **_File Path Check_**: `data/master_data`,  `data/master_data/labels/label.csv` and `data/preprocessedFinal_data`

To run the notebook Image Preprocessing.ipynb, execute the below command from the parent directory

```bash
jupyter notebook Image Preprocessing.ipynb
```
Files for reference: `image_preprocessing.ipynb` and `Image_Preprocessing.py`.

- By now we should have all the preprocessed image in **preprocessedFinal_data** folder

## Object Detection 
> Follow the [Object Detection](#yolov5-model-setup-and-training) section to run this step

## CNN Model

- Dynamic Image Preprocessing
  - Image Resizing, Decoding, Normalizing
- Model Building
- Model Compilation and Fitting
- Model Evaluation
- Exporting model variable

Open the notebook ModelBuilding.ipynb

> **_File Path Check_**: `data/master_data/labels/label.csv` and `data/preprocessedFinal_data`

To run the notebook ModelBuilding.ipynb, execute the below command from the parent directory

```bash
jupyter notebook ModelBuilding.ipynb
```
Files for reference: `ModelBuilding copy.ipynb`

- By now we should have a model variable (`final_CNN_model.h5`) downloaded in the main parent folder


## User Interface Setup

A web application is developed using `streamlit`

### Folder Creations
There are certain folders that need to be created in the appropriate locations.
- Create a folder named ***tests*** inside your yolov5 folder and store all the images that you need to test.
- Create two folders in the parent directory, named ***preprocessed*** and ***detections***

>**_File Path Check_**: Make sure to replace the correct paths of yolov5 model variable (`yolov5/runs/train/yolov5s_results/best.pt`) and CNN model variable (`main parent folder`) in `app.py` script


## YOLOV5 MODEL SETUP AND TRAINING

1. Clone the YOLOv5 Repository
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```
2. Install all the required dependencies for yolov5 model setup
```bash
pip install -r requirements.txt
```

### Data Folder Setup

1. Duplicate the `master_data` folder inside the `data/` folder and rename it as `yolo_data`
2. Follow the below steps for both `train` and `validate` folder inside `yolo_data` folder
   - Create two folders `images` and `labels` inside both `train` and `validate` folders
   - Move all the images from `train/` and `validate/` folders to their respective `images` folder 
3. Delete the `test` folder completely


To run the file YoloData_SetupScript.py, execute the below command from the parent directory
```bash
python YoloData_SetupScript.py
```
The above script is to automate the following:
- `Labels` folder in both `train` and `validate` folders should have text files that contains the corresponding bounding box coordinates of each image
- Each text file inside `labels` folder should be named with the corresponding image names

Files for reference: `detect.ipynb`

- By now we should have all the rearranged image in **yolo_data/(train and validate)/images** folder and all the labels in **yolo_data/(train and validate)/labels** folder

### Configuration

**Instructions:** 

> Change the file paths of ***yolo_data/(train and validate)/images*** in `data.yaml` accordingly.

> Check for the presence of  `yolov5s-seg.pt` variable inside the parent directory


### Training

Training a **yolov5s** model on a custom dataset with a pre-trained weight (`yolov5s-seg.pt`)

To initiate the training process, execute the below command from the parent directory

```bash
python yolov5/train.py --img 320 --batch 16 --epochs 25 --data data.yaml --cfg yolov5s.yaml --weights yolov5s-seg.pt --name yolov5s_results
```
- By now we should have a trained model weight in `runs/train/yolov5s_results/` folder 

> Get back to [CNN Model](#cnn-model) section to proceed further.

## USAGE

By executing the `app.py` file, the Developed User Interactive Application is launched to showcase the project. 

The file contains the final execution pipepline which comprises of all the following process:
- All the Image Preprocessing steps
- Object Detection using the trained yolov5 model variable 
- Model Prediction using the trained CNN model variable

To run the `app.py` file and launch the UI, execute the following command from the parent directory:

> Make sure to activate your environment
- Mac/unix systems: `source Venv\bin\activate`
- Windows systems: `Venv\Scripts\activate`

```bash
streamlit run app.py
```

Model variables for usage: `final_CNN_model.h5` and `best.pt`

- By now you should be able see an User Interactive Application where you are prompted to upload an Image for testing.

## SYSTEM ARCHITECTURE
The system architecture is designed to facilitate a streamlined flow from data collection to model training and prediction. It includes:

- **Dataset Preparation:** Handling and merging of data, balancing classes.
- **Image Preprocessing:** Steps such as resizing, denoising, and contrast enhancement.
- **Object Detection:** Post-training detection and bounding box plotting on new images. 
- **Model Training:** Utilizes a three-layered CNN and the YOLOv5 model for object detection.


![SYSTEM ARCHITECTURE](images%20and%20videos/archietecture.jpeg)


## INTERFACE

![INTERFACE](images%20and%20videos/interface.jpeg)


## Contributors
- Nithish Kumar Senthil Kumar
- Subhiksha Murugesan
- Rishikesh Ramesh
- Rishi Manohar
- Nagul Pandian