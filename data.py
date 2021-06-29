from __future__ import print_function
from skimage.util.dtype import img_as_float32
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.util import img_as_uint
from pathlib import Path
import cv2
from sklearn.utils import class_weight
from PIL import Image
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from data import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


class Data():
    Sky = [128, 128, 128]  # gray
    Building = [128, 0, 0]  # red
    Pole = [192, 192, 128]  # bege
    Road = [128, 64, 128]  # purple
    Pavement = [60, 40, 222]  # blue
    Tree = [128, 128, 0]  # Olive
    SignSymbol = [192, 128, 128]  # almost red
    Fence = [64, 64, 128]  # blue
    Car = [64, 0, 128]  # dark purple
    Pedestrian = [64, 64, 0]  # green or yellow, I dont know
    Bicyclist = [0, 128, 192]  # dark washed azure
    Unlabelled = [0, 0, 0]  # black

    COLOR_DICT = np.array(
        [
            Sky,
            Building,
            Pole,
            Road,
            Pavement,
            Tree,
            SignSymbol,
            Fence,
            Car,
            Pedestrian,
            Bicyclist,
            Unlabelled,
        ]
    )

    @staticmethod
    def adjust_data(img, mask, flag_multi_class, num_class):
        if flag_multi_class:
            img = img / 255
            mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
            new_mask = np.zeros(mask.shape + (num_class,))
            for i in range(num_class):
                # for one pixel in the image, find the class in mask and convert it into one-hot vector
                # index = np.where(mask == i)
                # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
                # new_mask[index_mask] = 1
                new_mask[mask == i, i] = 1
            new_mask = (
                np.reshape(
                    new_mask,
                    (
                        new_mask.shape[0],
                        new_mask.shape[1] * new_mask.shape[2],
                        new_mask.shape[3],
                    ),
                )
                if flag_multi_class
                else np.reshape(
                    new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
                )
            )
            mask = new_mask
        elif np.max(img) > 1:
            img = img.astype('float32')
            img = img / 255

            mask = mask.astype('float32')
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

            # get_class_weights(mask)

        # dpi = 80
        # figsize = 416 / float(dpi), 320 / float(dpi)
        # plt.figure(figsize=figsize)
        # plt.imshow(img[0,:,:,:])
        # plt.show()

        return (img, mask)

    @staticmethod
    def data_generator(
        batch_size,
        data_path,
        image_folder,
        mask_folder,
        aug_dict,
        image_color_mode="rgb",
        mask_color_mode="rgb",
        image_save_prefix="image",
        mask_save_prefix="mask",
        flag_multi_class=False,
        num_class=2,
        save_to_dir=None,
        target_size=(256, 256),
        seed=1,
        class_mode=None,
    ):
        """
        can generate image and mask at the same time
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        """
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        image_generator = image_datagen.flow_from_directory(
            data_path,
            classes=[image_folder],
            class_mode=class_mode,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed,
            save_format='jpg'
        )
        mask_generator = mask_datagen.flow_from_directory(
            data_path,
            classes=[mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed,
            save_format='jpg'
        )

        generator = zip(image_generator, mask_generator)
        for (img, mask) in generator:
            img, mask = Data.adjust_data(img, mask, flag_multi_class, num_class)
            yield (img, mask)

    @staticmethod
    def test_generator(
        path,
        ext=".JPG",
        num_image=30,
        target_size=(256, 256),
        flag_multi_class=False,
        as_gray=False,
    ):
        imgs = glob.glob(path + "/*" + ext)
        
        for item in imgs:
            # os.path.join(test_path,"%d.jpg"%i)
            img = io.imread(item, as_gray=as_gray)
            img = img.astype('float32')
            img = img / 255

            # w = img.shape[1]
            # h = img.shape[0]
            # c = img.shape[2]
            # inverted = (w,h,c)

            img = trans.resize(img, target_size)
            img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
            img = np.reshape(img, (1,) + img.shape)

            yield img 

        # for item in enumerate(imgs):
        #     img = io.imread(item[1], as_gray = as_gray)
        #     img = img / 255.
        #     img = trans.resize(img, target_size)
        #     img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        #     img = np.reshape(img,(1,)+img.shape)
        #     yield img

    @staticmethod
    def gene_data_npy(
        data_path,
        flag_multi_class=False,
        num_class=2,
        image_folder="images",
        mask_folder="masks",
        image_as_gray=False,
        mask_as_gray=False,
    ):
        image_path = os.path.join(data_path, image_folder)
        mask_path = os.path.join(data_path, image_folder)

        image_name_arr = glob.glob(os.path.join(image_path, "*.JPG"))
        image_arr = []
        mask_arr = []

        # fp_images: np.memmap = None
        # fp_masks: np.memmap = None

        for index, item in enumerate(image_name_arr):
            img = io.imread(item, as_gray=image_as_gray)
            img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
            mask = io.imread(
                item.replace(image_path, mask_path).replace(image_folder, mask_folder),
                as_gray=mask_as_gray,
            )
            mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
            img, mask = Data.adjust_data(img, mask, flag_multi_class, num_class)
            image_arr.append(img)
            mask_arr.append(mask)

            if index % 50 == 0:
                print(f"generate data - {index}/{len(image_name_arr)}")

            # if(index > 0 and index % 5 == 0):
            #     print(f'saving data... {index}/{len(image_name_arr)}')

            #     if(fp_images == None or not fp_images.all()):
            #         fp_images = np.memmap(f"{data_path}aug/images_arr.mymemmap", dtype='float32', mode='w+', shape=(np.shape(image_arr)))

            #     if(fp_masks == None or not fp_masks.all()):
            #         fp_masks = np.memmap(f"{data_path}aug/masks_arr.mymemmap", dtype='float32', mode='w+', shape=(np.shape(mask_arr)))

            #     fp_images[:] = image_arr[:]
            #     fp_masks[:] = mask_arr[:]

            #     image_arr.clear()
            #     mask_arr.clear()

        image_arr = np.array(image_arr)
        mask_arr = np.array(mask_arr)

        return image_arr, mask_arr

    @staticmethod
    def label_visualize(num_class, color_dict, img):
        img = img[:, :, 0] if len(img.shape) == 3 else img
        img_out = np.zeros(img.shape + (3,))
        for i in range(num_class):
            img_out[img == i, :] = color_dict[i]

        img_out = img_out.astype('float32')
        return img_out / 255

    @staticmethod
    def save_result(save_path, npyfile, imgs, flag_multi_class=False, num_class=2):

        Path(save_path).mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(npyfile):
            
            if(i % 500 == 0):
                print(f'print {i}: {item}')

            if flag_multi_class:
                img = Data.label_visualize(num_class, Data.COLOR_DICT, item)
                io.imsave(os.path.join(save_path, imgs[i] + "_predict.jpg"), img)
            else:
                img = item[:, :, 0]
                img = img.astype('float32')
                img = img / 255
                img[img > 0.50] = 1
                img[img <= 0.50] = 0
                io.imsave(os.path.join(save_path, imgs[i] + "_predict.jpg"), img_as_float32(img)) 

            # img[img > 0.50] = 1
            # img[img <= 0.50] = 0 
            
        
