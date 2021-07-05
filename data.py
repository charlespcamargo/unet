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
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


class Data():
    sky = [128, 128, 128]  # gray
    building = [128, 0, 0]  # red
    pole = [192, 192, 128]  # bege
    road = [128, 64, 128]  # purple
    pavement = [60, 40, 222]  # blue
    tree = [128, 128, 0]  # Olive
    sign_symbol = [192, 128, 128]  # almost red
    fence = [64, 64, 128]  # blue
    car = [64, 0, 128]  # dark purple
    pedestrian = [64, 64, 0]  # green or yellow, I dont know
    bicyclist = [0, 128, 192]  # dark washed azure
    unlabelled = [0, 0, 0]  # black
    predict_suffix = '_predict.png'

    COLOR_DICT = np.array(
        [
            sky,
            building,
            pole,
            road,
            pavement,
            tree,
            sign_symbol,
            fence,
            car,
            pedestrian,
            bicyclist,
            unlabelled,
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
        imgs,
        path,
        ext=".JPG",
        num_image=30,
        target_size=(256, 256),
        flag_multi_class=False,
        as_gray=False
    ): 
        for item in imgs:
            img = io.imread(path + item, as_gray=as_gray)
            img = img.astype('float32')
            img = img / 255 
            img = trans.resize(img, target_size)
            img = np.reshape(img, (1,) + img.shape)

            yield img        

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

            if flag_multi_class:
                img = Data.label_visualize(num_class, Data.COLOR_DICT, item)
                io.imsave(os.path.join(save_path, imgs[i] + Data.predict_suffix), img)
            else:
                img = item[:, :, :]
                prediction_binary = np.abs(np.mean(img, axis=2) > 0.5) * 255
                prediction_binary = prediction_binary.astype(np.uint8)
                io.imsave(os.path.join(save_path, imgs[i] + Data.predict_suffix), prediction_binary)
    
    @staticmethod
    def compare_result(base_path, imgs, font_path = 'assets/arial-unicode.ttf'):
        
        save_path = base_path + "diff/"    
        Path(save_path).mkdir(parents=True, exist_ok=True)
        font = ImageFont.truetype(font_path, 18)
        gsd = 1.73 # 1.73cm/px

        for i, file_name in enumerate(imgs):            
            image_data = Data.load_images_to_compare(base_path, file_name)
            images = image_data[0]
            images_masks_size = image_data[1]
            border = columns = 4
            subtitle_size = 20
            rows = 2
            total_width, max_height, border, img_mais_larga, img_mais_alta = Data.get_sizes(images, border, columns, rows, subtitle_size)
            
            composed_img = Image.new('RGBA', (total_width, max_height), color="gray")
            x_offset = border
            y_offset = border + subtitle_size
            total_imgs = (len(images) if rows * columns  > len(images)  else rows * columns) 
            current_row = 0
            current_column = 0

            for j in range(0, total_imgs):
                current_column += 1
                img = images[j]
                mask_size = images_masks_size[j]
                Data.add_image(composed_img, img, current_column, x_offset, y_offset, subtitle_size, border, font, mask_size, gsd)
                x_offset += (img_mais_larga + border)

                if(current_column == columns):
                    current_column = 0
                    current_row += 1
                    x_offset = border
                    y_offset = (subtitle_size + border) * (current_row + 1) + img_mais_alta                

            composed_img.save(os.path.join(save_path, imgs[i] + ".png"))

            # if(i >= 5):
            #     break

    @staticmethod
    def load_images_to_compare(base_path, file_name):
        
        original = io.imread(os.path.join(base_path, "images", file_name), as_gray=False)
        mask = io.imread(os.path.join(base_path, "masks", file_name), as_gray=False)
        predict = io.imread(os.path.join(base_path, "results", file_name + Data.predict_suffix), as_gray=False)
        
        mask[mask >= 128] = 255
        mask[mask < 128] = 0
        mask_predict_diff = ((np.abs(mask[:,:,0] - predict) > 0.5) * 255).astype(np.uint8)
        
        original_rgba = Image.fromarray(original).convert("RGBA")
        original_filter_mask, mask_pixels = Data.parse_mask_to_rgba(mask, 0)
        original_filter_predict, predict_mask_pixels = Data.parse_mask_to_rgba(predict, 1)
        original_filter_diff, diff_mask_pixels = Data.parse_mask_to_rgba(mask_predict_diff, 2)
        original = Image.fromarray(original)

        image_data = [
                        [
                            original,
                            Image.fromarray(mask).convert('RGB'),
                            Image.fromarray(predict).convert('RGB'),
                            Image.fromarray(mask_predict_diff).convert('RGB'),
                            original,
                            Image.alpha_composite(original_rgba, original_filter_mask),
                            Image.alpha_composite(original_rgba, original_filter_predict),
                            Image.alpha_composite(original_rgba, original_filter_diff),
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            mask_pixels,
                            predict_mask_pixels,
                            diff_mask_pixels
                        ]
                    ]

        return image_data

    @staticmethod
    def get_sizes(images, border, columns, rows, subtitle_size):
        widths, heights = zip(*(img.size for img in images))
        img_mais_alta = max(widths)
        img_mais_larga = max(heights)
        total_width = (img_mais_alta * columns) + ((columns + 1) * border)
        max_height = (img_mais_larga * rows) + ((rows + 1) * (border + subtitle_size))

        return total_width, max_height, border, img_mais_larga, img_mais_alta

    @staticmethod
    def add_image(composed_img, img, column_type, x_offset, y_offset, subtitle_size, border, font, mask_size, gsd):
        composed_img.paste(img, (x_offset, y_offset))
        draw = ImageDraw.Draw(composed_img)
        subtitle_text = Data.get_image_subtitle_text(column_type, mask_size, gsd)
        draw.text((x_offset, y_offset - subtitle_size - border), subtitle_text, (255, 255, 255), font=font)

    @staticmethod
    def get_image_subtitle_text(column_type, mask_size, gsd):
        if(column_type == 1):
            return f'original{Data.get_size(mask_size, gsd)}'
        elif (column_type == 2):
            return f'ground-truth{Data.get_size(mask_size, gsd)}'
        elif (column_type == 3):
            return f'predict{Data.get_size(mask_size, gsd)}'
        else:
            return f'diff ground-truth and predict{Data.get_size(mask_size, gsd)}'

    @staticmethod
    def get_size(mask_size, gsd):
        if(mask_size == 0 or gsd == 0):
            return ''

        total = mask_size * gsd

        if(total / 100 > 1):
            return f': {mask_size}px - {round(total / 10000, 2)}m²' 
        else:
            return f': {mask_size}px - {round(total, 2)}cm' 

    @staticmethod
    def parse_mask_to_rgba(mask, mask_type):
        mask = Image.fromarray(mask)
        if(mask.mode != 'RGBA' and mask.mode != 'RGB'):
            mask = mask.convert('RGB')

        RGBA = np.dstack((mask, np.zeros(mask.size, dtype = np.uint8) + 255))            

        # Make mask of black or white pixels too. - mask is True where image has the right color
        black_mask = (RGBA[:, :, 0:3] == [0,0,0]).all(2)
        white_mask = (RGBA[:, :, 0:3] == [255,255,255]).all(2)
        
        # Make all pixels matched by mask into transparent ones
        RGBA[black_mask] = (0,0,0,0)

        # Make all pixels matched by mask into "filter" purple color with transparency
        alpha_percent = 50
        alpha = int(alpha_percent/100*255)

        if(mask_type == 0):
            color = (255, 131, 0, alpha) # purple
        elif(mask_type == 1):            
            color = (110, 0, 110, alpha) # purple
        else:
            alpha_percent = 75
            alpha = int(alpha_percent/100*255)
            color = (255, 0, 0, alpha)  # red

        RGBA[white_mask] = color

        return Image.fromarray(RGBA), np.sum(white_mask)
