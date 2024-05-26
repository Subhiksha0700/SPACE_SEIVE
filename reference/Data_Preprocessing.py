# Import statements
import pandas as pd
import numpy as np

import os
import shutil

import zipfile


# Create a folder with a tree structure:
    # master_data
        # |- train
        # |- val
        # |- test


# to find different types of file extensions in a folder
#find . -type f -exec sh -c 'echo "${0##*.}"' {} \; | sort | uniq -c

# command for deleting faster in mac
# find . -maxdepth 1 -name "*.jpg" -exec rm {} \;    # go to the respective directory

# Check if the destination folder is not empty
if os.path.exists('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/'):
    # Remove the entire directory and its contents
    shutil.rmtree('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/')

#------------------------------CODE FOR PREPROCESSING 2021 DATASET-----------------------------------#

train_data = pd.read_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/ICIP-2021/labels/train_labels.csv')
val_data = pd.read_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/ICIP-2021/labels/validate_labels.csv')
test_data = pd.read_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/ICIP-2021/labels/test_labels.csv')

# Dropping unwanted columns
val_data.drop(['id', 'depth'], inplace = True, axis = 1)
train_data.drop(['id', 'depth'], inplace = True, axis = 1)
test_data.drop(['id', 'depth'], inplace = True, axis = 1)

# Function to check value_counts
def valueCounts(data, lst):
    for col in lst:
        print(col,'\n')
        print(data[col].value_counts())
        print('\n')

valueCounts(test_data, ['class'])
valueCounts(train_data, ['class'])
valueCounts(val_data, ['class'])

# It is confirmed that class `6` is debris in test dataset


# Class for Ordinal encoding - classes
order = {'CubeSat': 1,
 'Jason': 2,
 'TRMM': 3,
 'Cloudsat': 4,
 'Terra': 5,
 'Debris': 6,
 'Sentinel-6': 7,
 'AcrimSat': 8,
 'Aura': 9,
 'Aquarius': 10,
 'Calipso': 11}

# Preserving the original class names
train_data['class1'] = train_data['class'].copy()
val_data['class1'] = val_data['class'].copy()
test_data['class1'] = test_data['class'].copy()

# Ordinal encoding - classes
train_data['class'] = train_data['class'].map(order)
val_data['class'] = val_data['class'].map(order)


# Function that takes a random sample from each class of satillites(apart from debris) and merged with the filtered dataset of debris
# END GOAL OF THE FUNCTION: To have balanced no of records between debris and satellites
def stratifiedSampling(data, top):
    print(data.shape)
    WOdebris_data = data[data['class'] != 6]
    WOdebris_sampLst = []
    
    for col in WOdebris_data['class'].unique().tolist():
        
        col_temp_data = WOdebris_data[WOdebris_data['class'] == col].copy()
        
        col_temp_data = col_temp_data.sample(frac=1, random_state=102)
        col_temp_data = col_temp_data.head(top)
        WOdebris_sampLst.extend(col_temp_data.index.tolist())
    
    nonDebri_data = data.loc[WOdebris_sampLst]
    Debri_data = data[data['class'] == 6]
    
    Filtered_data = pd.concat([Debri_data, nonDebri_data], ignore_index=True)
    Filtered_data = Filtered_data.sample(frac=1, random_state=102)
    print(Filtered_data.shape)
    
    return Filtered_data


Filtered_train = stratifiedSampling(train_data, 1500)
Filtered_val = stratifiedSampling(val_data, 500)


# Label encoding with 0 (debris) and 1 (all the other satellites)
Filtered_train['class'] = Filtered_train['class'].apply(lambda x: 0 if x == 6 else 1)
Filtered_val['class'] = Filtered_val['class'].apply(lambda x: 0 if x == 6 else 1)
test_data['class'] = test_data['class'].apply(lambda x: 0 if x == 6 else 1)


# Appending satellite name and year to original image name, changing file extension and relocating it
def imageRenameRelocalize(data, tag):
    data['image_c'] = data['image'].str[:-7]
    data['image'] = data['image'].str[:-4]
    data['image'] = data['image'] + '.jpg'
    
    # counters for checking non-existing images
    found_count = 0
    not_found_count = 0
    
    for clas in data['class1'].unique().tolist():
        data.loc[data['class1'] == clas,'image_c'] = data[data['class1'] == clas]['image_c'] + clas + '_2021.jpg'
        for index, row in data[data['class1'] == clas].iterrows():
            
            actual_image_name = row['image']
            changing_image_name = row['image_c']

            original_image_path = os.path.join('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/ICIP-2021/' + tag + '_rgb/' + clas, actual_image_name)
            destination_image_path = os.path.join('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/' + tag, changing_image_name)
    
            try:
                # Copy the image file to another location
                shutil.copy(original_image_path, destination_image_path)
                found_count += 1
            except FileNotFoundError:
                not_found_count += 1
                data.drop([index], inplace = True, axis = 0)         

    print(tag, f"Files found: {found_count}")
    print(tag, f"Files not found: {not_found_count}")
    
    return data

Filtered_train = imageRenameRelocalize(Filtered_train, 'train')
Filtered_val = imageRenameRelocalize(Filtered_val, 'val')



# Appending year to original image name, changing file extension and relocating test dataset
def testImageRelocalize():
    test_data['image_c'] = test_data['image'].str[:-7]
    test_data['image'] = test_data['image'].str[:-4]
    test_data['image'] = test_data['image'] + '.jpg'

    found_count = 0
    not_found_count = 0
    
    test_data.loc[:,'image_c'] = test_data['image_c'] + '2021.jpg'
    for index, row in test_data.iterrows():
        
        actual_image_name = row['image']
        changing_image_name = row['image_c']

        original_image_path = os.path.join('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/ICIP-2021/test_rgb', actual_image_name)
        destination_image_path = os.path.join('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/test', changing_image_name)
    
        try:
            # Copy the image file to another location
            shutil.copy(original_image_path, destination_image_path)
            found_count += 1
        except FileNotFoundError:
            not_found_count += 1
            test_data.drop([index], inplace = True, axis = 0)

    print(f"Files found: {found_count}")
    print(f"Files not found: {not_found_count}")   

testImageRelocalize()



Filtered_train.drop(['class1', 'image'], inplace = True, axis = 1)
Filtered_val.drop(['class1', 'image'], inplace = True, axis = 1)
test_data.drop(['class1', 'image'], inplace = True, axis = 1)

Filtered_train['usage'] = 'train'
Filtered_val['usage'] = 'val'
test_data['usage'] = 'test'

Filtered_train['year'] = 2021
Filtered_val['year'] = 2021
test_data['year'] = 2021

test_data['illumination'] = np.nan

# Define the desired order of column names
desired_order = ['illumination', 'image_c', 'bbox', 'year', 'usage', 'class']

# Reorder the columns of the DataFrame
Filtered_train = Filtered_train[desired_order]
Filtered_val = Filtered_val[desired_order]
test_data = test_data[desired_order]

# Combining train, val and test
data_2021 = pd.concat([Filtered_train, Filtered_val, test_data], ignore_index=True)
data_2021.rename(columns={'image_c': 'filename'}, inplace=True)
data_2021.reset_index(drop=True, inplace=True)

# Exporting 2021 dataset
data_2021.to_csv('data/data_2021.csv', index=False)




#------------------------------CODE FOR PREPROCESSING 2022 DATASET-----------------------------------#

train = pd.read_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/spark-2022-stream-1/labels/train.csv')
test = pd.read_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/spark-2022-stream-1/labels/sample_submission.csv')
val = pd.read_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/spark-2022-stream-1/labels/val.csv')


# Function that takes a random sample from each class of satillites(apart from debris) and merged with the filtered dataset of debris
# END GOAL OF THE FUNCTION: To have balanced no of records between debris and satellites
def filter_train_val(data,flag):
    new_data = data[data['class']=='debris']
    list = data['class'].value_counts().index.tolist()
    list.remove('debris')
    
    class_df = pd.DataFrame()
    for class_name in list:
        if flag == 'train':
            Class = data[data['class'] == class_name].head(600)
        else:
            Class = data[data['class'] == class_name].head(200)
        class_df = pd.concat([class_df, Class])
    
    class_df = class_df.sample(frac=1, random_state=42)
    class_df.reset_index(drop=True, inplace=True)
    
    Filtered_data = pd.concat([new_data, class_df])
    Filtered_data = Filtered_data.sample(frac=1, random_state=42)
    Filtered_data.reset_index(drop=True, inplace=True)
    return Filtered_data


Filtered_train  = filter_train_val(train,'train')
Filtered_val = filter_train_val(val,'val')


# Appending year to original image name and changing file extension
Filtered_train['filename'] = Filtered_train['filename'].str[:-4]
Filtered_train['filename'] = Filtered_train['filename'] + '_2022.jpg'

Filtered_val['filename'] = Filtered_val['filename'].str[:-4]
Filtered_val['filename'] = Filtered_val['filename'] + '_2022.jpg'


# Label encoding with 0 (debris) and 1 (all the other satellites)
category_to_zero = 'debris'
Filtered_train['class'] = Filtered_train['class'].apply(lambda x: 0 if x == category_to_zero else 1)
Filtered_val['class'] = Filtered_val['class'].apply(lambda x: 0 if x == category_to_zero else 1)


# Relocating Images
def image_transfer(data,flag):
    filenames_data_df = pd.DataFrame() 
    filenames_data_df['filename'] = data['filename']
    
    if (flag == 'train'):
        folder_2022_train = "/Users/nithish/Documents/Academics/Spring 24/AML/project/data/spark-2022-stream-1/train"
        master_train_folder = "/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/train"
    else:
        folder_2022_val = "/Users/nithish/Documents/Academics/Spring 24/AML/project/data/spark-2022-stream-1/val"
        master_val_folder = "/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/val"
    
    for filename in filenames_data_df['filename']:    
        finalName = filename
        filename = filename[:-9]
        filename = filename +'.jpg'
        
        if (flag == 'train'):
            source_path = os.path.join(folder_2022_train, filename)
            destination_path = os.path.join(master_train_folder, finalName)
        else:
            source_path = os.path.join(folder_2022_val, filename)
            destination_path = os.path.join(master_val_folder, finalName)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
    
    print("Images transferred.")


image_transfer(Filtered_train,'train')
image_transfer(Filtered_val,'val')


Filtered_train['usage'] = 'train'
Filtered_val['usage'] = 'val'
test['usage'] = 'test'

Filtered_train['year'] = 2022
Filtered_val['year'] = 2022
test['year'] = 2022

Filtered_train['illumination'] = np.nan
Filtered_val['illumination'] = np.nan
test['illumination'] = np.nan
test['bbox'] = np.nan
test['class'] = np.nan

# Define the desired order of column names
desired_order = ['illumination', 'filename', 'bbox', 'year','usage','class']

# Reorder the columns of the DataFrame
Filtered_train = Filtered_train[desired_order]
Filtered_val = Filtered_val[desired_order]
test = test[desired_order]

# Combining train, val and test
data_2022 = pd.concat([Filtered_train, Filtered_val, test], ignore_index=True)
data_2022.reset_index(drop=True, inplace=True)

# Exporting 2021 dataset
data_2022.to_csv('data/data_2022.csv', index=False)



#------------------------------MERGING TWO DATASETS 2021 AND 2022---------------------------------------#

concatenated_df = pd.concat([data_2021, data_2022], ignore_index=True)

concatenated_df.to_csv('/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data/labels.csv', index=False)



#------------------------------ZIPPING MASTER FOLDER---------------------------------------#

def zip_folder(folder_path, zip_path):

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

folder_to_zip = '/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data'  # Path to the folder to be zippe
zip_file_path = '/Users/nithish/Documents/Academics/Spring 24/AML/project/data/master_data.zip' # Path to the resulting zip file

zip_folder(folder_to_zip, zip_file_path)
