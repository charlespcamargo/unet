from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.util import img_as_float, img_as_ubyte 
from pathlib import Path

from sklearn.utils import class_weight

Sky = [128,128,128] # gray
Building = [128,0,0] # red
Pole = [192,192,128] # bege
Road = [128,64,128] # purple
Pavement = [60,40,222] # blue
Tree = [128,128,0] # Olive
SignSymbol = [192,128,128] # almost red
Fence = [64,64,128] # blue
Car = [64,0,128] # dark purple 
Pedestrian = [64,64,0]  # green or yellow, I dont know
Bicyclist = [0,128,192] # dark washed azure
Unlabelled = [0,0,0] # black

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjust_data(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def data_generator(batch_size, data_path, image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1, class_mode = None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        data_path,
        classes = [image_folder],
        class_mode = class_mode,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        data_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    X = []
    Y = []
    data_generator = zip(image_generator, mask_generator)
    for (img,mask) in data_generator:
        img,mask = adjust_data(img,mask,flag_multi_class,num_class)
        X.append(img)
        Y.append(mask)

    class_weights = class_weight.compute_class_weight('balanced', 
                                                       np.unique(Y), 
                                                       Y)
    return (X, Y, class_weights)

def test_generator(path, ext = '.JPG', num_image = 30, target_size = (256,256), flag_multi_class = False, as_gray = True):    
    imgs = glob.glob(path + '/*' + ext)
    for item in imgs:
        #os.path.join(test_path,"%d.jpg"%i)
        img = io.imread(item, as_gray = as_gray)
        img = img / 255.
        img = trans.resize(img, target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)               
        yield img

def gene_train_npy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.jpg"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_grey = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjust_data(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def label_visualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def save_result(save_path, npyfile, imgs, flag_multi_class = False,num_class = 2):
    
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for i,item in enumerate(npyfile):
        #if flag_multi_class:
        #    img = label_visualize(num_class,COLOR_DICT,item)
        #else:            
        img=item[:,:,0] 
        
        img[img>0.5] = 1
        img[img<=0.5] = 0
        
        io.imsave(os.path.join(save_path, imgs[i] + "_predict.png"), img_as_ubyte(img))

