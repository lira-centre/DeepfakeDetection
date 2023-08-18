"""
Please cite:
    
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). 
Neural Networks, 2020 
"""

#import the libraries

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor



model_VT = torchvision.models.vit_l_32(weights='DEFAULT')
feature_extractor = nn.Sequential(*list(model_VT.children())[:-1])
encoder = feature_extractor[1]

 
def extractor(img_path):

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),# VGG-16 Takes 224x224 images as input, so we resize all of them
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    try:
 
        img = Image.open(img_path)
        #print(img.getbands())
        
        if img.getbands() == ('L',):
            return
        


        img = transform(img)

        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
   

        # For pre-trained weights
        n = x.shape[0]
        x = model_VT._process_input(x)
        batch_class_token = model_VT.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        y = encoder(x)

        y = y[:, 0]


        y = torch.squeeze(y)
        y = torch.flatten(y)
        y = y.data.numpy()

        return y
    
    except Exception as e:
        return


#Load the data directory  where the images are stored
data_dir = 'train_real_fake\\fake\\'

contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]


enum_classes = list(enumerate(classes,0))


images = []
batch = []
labels = []

dataset = None

def processFolders(each):
    global dataset
    print("Starting {} images".format(each[1]))
    class_path = data_dir + each[1]
    files = os.listdir(class_path)
    print(type(files))

    ii=0

    for ii, file in enumerate(files, 1):

        
        img = os.path.join(class_path, file)
        data = []

        # For fake images
        file_name_type = 'fake_'+each[1]+'_'+file

        # For original images
        #file_name_type = 'original_'+each[1]+'_'+file

        data.append(file_name_type)
  
        ii+=1

        # For fake images (class 1), comment for original images
        data.append(str(1))
        # For original images (class 0), comment for fake images
        #data.append(str(0))

        data = np.array(data)
        features = extractor(img)  # Extract features
        if features is not None:
            temp = np.array([features, data], dtype="O")
            if dataset is not None:
                dataset = np.vstack((dataset, temp))
            else:
                dataset = temp
            
        if ii % 50 == 0:
            print("finish {} for class {}".format(ii,each[0]))



if __name__ == '__main__':

    import time
    start = time.time()

    # Increase threads for larger datasets
    with ThreadPoolExecutor(max_workers=1) as executor:
        for result in executor.map(processFolders, enum_classes):
            pass

    end = time.time()
    print(end-start)


    np_batch = np.stack(dataset[:,0])
    np_info = np.stack(dataset[:,1])
    np_labels = np_info[:,1]
    np_images = np_info[:,0]

    np_labels_T = np_labels.reshape(-1,1)
    np_images_T = np_images.reshape(-1,1)
    print(np_labels_T)

    np_images_labels = np.hstack((np_images_T,np_labels_T))
    print(np_images_labels)


    #Slpit the data into training and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
    np_batch, np_images_labels, test_size=0.1, random_state=0)

    #Convert data to Pandas in order to save as .csv
    import pandas as pd
    data_df_X_train = pd.DataFrame(X_train)
    data_df_y_train = pd.DataFrame(y_train)
    data_df_X_test = pd.DataFrame(X_test)
    data_df_y_test = pd.DataFrame(y_test)

    print(data_df_X_train)


    # Save file as .csv
    data_df_X_train.to_csv('data_df_X_train_deepfake_ffhq_finetuned.csv',mode='a',header=False,index=False)
    data_df_y_train.to_csv('data_df_y_train_deepfake_ffhq_finetuned.csv',mode='a',header=False,index=False)
    data_df_X_test.to_csv('data_df_X_test_deepfake_ffhq_finetuned.csv',mode='a',header=False,index=False)
    data_df_y_test.to_csv('data_df_y_test_deepfake_ffhq_finetuned.csv',mode='a',header=False,index=False)


