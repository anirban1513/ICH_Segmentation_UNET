#from processing import *
from modelcode import *
import shutil
from pathlib import Path
import tensorflow
from tensorflow.keras.callbacks import CSVLogger

#importing the libraries
import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import numpy as np, os, random
from pathlib import Path
from imageio import imsave
import nibabel as nib
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib


#Swaping the values of No_hemorrage to Has Hemorrage
def swap_col_value(x):
    if x == 0:
        return 1
    else:
        return 0


#making Image ID
def get_image_fname(row):
        
    img_id = str(row['idx_num']) + '.png'
    
    return img_id
    




def get_image_pathfname(row):
        
    img_path = 'workdir/image/'+ str(row['idx_num']) + '.npy'
    
    return img_path
    
def get_mask_pathfname(row):
        
    mask_path = 'workdir/label/'+ str(row['idx_num']) + '.npy'
    
    return mask_path
    

#Stratified Split
def get_value_label(row):
        
    
    return np.sum(np.load('workdir/label/'+ str(row['idx_num']) + '.npy')/255)/(512**2) 
    

#making Coverage Class
def cov_to_class(val):    
    for i in range(0,13):
        if val*10 <= i :
            return int(i/max_value)

if __name__ == '__main__':

    #getting the root path of the directory
    root_path = os.getcwd()


    #Reading the csv file and saving it to a dataframe
    df_dmg = pd.read_csv('hemorrhage_diagnosis_raw_ct.csv', delimiter=',')
    df_dmg.dfname = 'hemorrhage_diagnosis_raw_ct.csv'

    #Swaping Values of No hemorrage & Has hemorage
    df_dmg['Has_Hemorrhage'] = df_dmg['No_Hemorrhage'].apply(swap_col_value)
    df_dmg = df_dmg.drop('No_Hemorrhage', axis=1)

    #Creating a column with Index numbers
    df_dmg["idx_num"] = [*range(0, 2814, 1)]

    # create a new column with Image file names
    df_dmg['img_id'] = df_dmg.apply(get_image_fname, axis=1)

    #Image Directory making
    numSubj = 82
    new_size = (512, 512)      
    window_specs=[40,120]      #Brain window
    currentDir = root_path
    datasetDir = str(Path(currentDir))

    # Reading labels
    hemorrhage_diagnosis_df = pd.read_csv(
        Path('hemorrhage_diagnosis_raw_ct.csv'))
    hemorrhage_diagnosis_array = hemorrhage_diagnosis_df._get_values

    # reading images
    train_path = Path('workdir')
    image_path = train_path / 'image'
    label_path = train_path / 'label'
    if not train_path.exists():
        train_path.mkdir()
        image_path.mkdir()
        label_path.mkdir()


    counterI = 0
    for sNo in range(0+49, numSubj+49):
        if sNo>58 and sNo<66: #no raw data were available for these subjects
            next        #Avoiding the missing CT Scan Numbers
        else:
            #Loading the CT scan
            ct_dir_subj = Path(datasetDir,'ct_scans', "{0:0=3d}.nii".format(sNo))
            ct_scan_nifti = nib.load(str(ct_dir_subj))
            ct_scan = ct_scan_nifti.get_fdata()
            ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])

            #Loading the masks
            masks_dir_subj = Path(datasetDir,'masks', "{0:0=3d}.nii".format(sNo))
            masks_nifti = nib.load(str(masks_dir_subj))
            masks = masks_nifti.get_fdata()
            idx = hemorrhage_diagnosis_array[:, 0] == sNo     
            sliceNos = hemorrhage_diagnosis_array[idx, 1]          #Slice_numbers
            NoHemorrhage = hemorrhage_diagnosis_array[idx, 7]      #No_hemorrage values columns
            if sliceNos.size!=ct_scan.shape[2]:
                print('Warning: the number of annotated slices does not equal the number of slices in NIFTI file!')

            for sliceI in range(0, sliceNos.size):
                # Saving the a given CT slice
                x = ct_scan[:,:,sliceI]
                np.save(image_path / (str(counterI) + '.npy'), x)

                # Saving the segmentation for a given slice
                segment_path = Path(masks_dir_subj,str(sliceNos[sliceI]) + '_HGE_Seg.jpg')
                x = masks[:,:,sliceI]
                np.save(label_path / (str(counterI) + '.npy'), x)
                counterI = counterI+1
    #Training Directory
    os.mkdir('training')
    os.mkdir('training/img')
    os.mkdir('training/mask')

    #Testing Directory
    os.mkdir('testing')
    os.mkdir('testing/img')
    os.mkdir('testing/mask')

    #validation Directory
    os.mkdir('validation')
    os.mkdir('validation/img')
    os.mkdir('validation/mask')

    # create a new column with mask file names
    df_dmg['img_path'] = df_dmg.apply(get_image_pathfname, axis=1)
    # create a new column with mask file names
    df_dmg['mask_path'] = df_dmg.apply(get_mask_pathfname, axis=1)

    # create a new column with mask file intensity
    df_dmg['mask_val'] = df_dmg.apply(get_value_label, axis=1)

    #Mapping Coverage in dataframe
    max_value= max(df_dmg['mask_val'])
    df_dmg["coverage_class"] = df_dmg.mask_val.map(cov_to_class)

    #Splitting into Train Test and Vaidation
    from sklearn.model_selection import train_test_split
    train,test = train_test_split(df_dmg, test_size=0.2, stratify=df_dmg.coverage_class, random_state= 1234)
    train,val = train_test_split(train, test_size=0.2, random_state= 1234)
    
    #Copying file from datadirectory to train, test,validation image and mask folders
    for index,row in train.iterrows():
        shutil.move(str(row['img_path']),'training/img')
        shutil.move(str(row['mask_path']),'training/mask')

    for index,row in test.iterrows():
        shutil.move(str(row['img_path']),'testing/img')
        shutil.move(str(row['mask_path']),'testing/mask')

    for index,row in val.iterrows():
        shutil.move(str(row['img_path']),'validation/img')
        shutil.move(str(row['mask_path']),'validation/mask')
    

    # Define constants
    SEED = 101
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_TEST = 4

    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    IMAGE_CHANNELS = 1
    IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


    data_dir_train = 'training'
    data_dir_train_image = os.path.join(data_dir_train, 'img')
    data_dir_train_mask = os.path.join(data_dir_train, 'mask')

    data_dir_test = 'testing'
    data_dir_test_image = os.path.join(data_dir_test, 'img')
    data_dir_test_mask = os.path.join(data_dir_test, 'mask')

    data_dir_val = 'validation'
    data_dir_val_image = os.path.join(data_dir_val, 'img')
    data_dir_val_mask = os.path.join(data_dir_val, 'mask')

    train_generator = DataGenerator(data_dir_train_image , data_dir_train_mask, batch_size = BATCH_SIZE_TRAIN,
                                dim = IMAGE_HEIGHT, to_fit=True, shuffle=True)
    val_generator =  DataGenerator(data_dir_val_image , data_dir_val_mask , batch_size= BATCH_SIZE_TEST, 
                               dim = IMAGE_HEIGHT,to_fit=True, shuffle=True)
    test_generator = DataGenerator(data_dir_test_image , data_dir_test_mask , batch_size= BATCH_SIZE_TEST, 
                               dim = IMAGE_HEIGHT,to_fit=True, shuffle=True)

    NUM_TRAIN = 1800
    NUM_TEST = 563
    NUM_VAL = 451
    NUM_OF_EPOCHS = 50

    EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
    EPOCH_STEP_VAL = NUM_VAL // BATCH_SIZE_TEST

    #!touch 'result.csv'   #For opening a csv file to store the result


    tensorflow.keras.callbacks.CSVLogger('result.csv', separator=",", append=False)
    csv_logger = CSVLogger('result.csv')

    model = unet(inputs= (IMAGE_HEIGHT,IMAGE_WIDTH,1),lr=3e-4)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=4, min_lr=0.001)

    history = model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=val_generator, 
                    validation_steps=EPOCH_STEP_VAL,
                    epochs=NUM_OF_EPOCHS,
                    callbacks=[csv_logger,reduce_lr]
                   )


    model.save(f'UNET-BrainCTSegbaseline_{IMAGE_HEIGHT}_{IMAGE_WIDTH}_04.h5')

