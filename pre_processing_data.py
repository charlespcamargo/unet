from PIL import Image
import os
import glob
import numpy as np
from pathlib import Path
import shutil


class PreProcessingData():

    @staticmethod
    def crop_images_in_tiles(data_path, train_folder, val_folder, test_folder, image_folder, mask_folder, w, h, threshold = 50, force_delete = False):
        if(train_folder):
            PreProcessingData.crop_image(data_path, train_folder, force_delete, image_folder, mask_folder , w, h, True, threshold)
        if(val_folder):
            PreProcessingData.crop_image(data_path, val_folder, force_delete, image_folder, mask_folder , w, h, True, threshold)
        if(test_folder):
            PreProcessingData.crop_image(data_path, test_folder, force_delete, image_folder, mask_folder , w, h, False, threshold)

        PreProcessingData.move_ignored_items(data_path)


    @staticmethod
    def crop_image(data_path, data_folder, force_delete, image_folder="images", mask_folder="masks", w=512, h=512, validate_class_to_discard = False, threshold = 20):
        image_path = os.path.join(data_path, data_folder, image_folder)
        mask_path = os.path.join(data_path, data_folder, image_folder)
        image_name_arr = glob.glob(os.path.join(image_path, "*.JPG"))
        discarded_img = 0
        not_discarded = 0

        arr_weigths = []
        discard_folder = False

        stop = PreProcessingData.create_split_dirs(image_path, force_delete)
        if(stop):
            print(f'The data splits was create previously, any data to generate[{image_path}]')

            print(f"mean(black, white): {np.mean(arr_weigths, axis=0)}\n")

            return

        for index, item in enumerate(image_name_arr):

            file_name = item
            mask_name = item.replace(image_path, mask_path).replace(
                image_folder, mask_folder
            )

            img = Image.open(file_name)
            mask = Image.open(mask_name)

            width, height = img.size
            frame_num = 1

            if(index % 100 == 0):
                print(f"{data_folder} - {index}/{len(image_name_arr)}")

            for col_i in range(0, width, w):
                for row_i in range(0, height, h):
                    
                    # garantir que a ultima imagem tenha a mesma largura
                    if(col_i + w > width):
                        col_i = col_i + (width % w - w)

                    # garantir que a ultima imagem tenha a mesma altura
                    if(row_i + h > height):
                        row_i = row_i + (height % h - h)

                    croped = img.crop((col_i, row_i, col_i + w, row_i + h))
                    croped_mask = mask.crop((col_i, row_i, col_i + w, row_i + h))
                    
                    if(validate_class_to_discard):
                        discard_folder, (blackpercent, whitepercent)  = PreProcessingData.should_I_discard(croped_mask, threshold)
                    
                        if(not discard_folder):
                            arr_weigths.append((blackpercent, whitepercent))

                        if(discard_folder):
                            discarded_img = discarded_img + 1
                        else:
                            not_discarded = not_discarded + 1               

                    croped.save(PreProcessingData.get_name(file_name, frame_num, discard_folder))                
                    croped_mask.save(PreProcessingData.get_name(mask_name, frame_num, discard_folder)) 
                    
                    frame_num += 1

        if(validate_class_to_discard):
            print(f'\n{data_folder} - discarded_img: {discarded_img} - not_discarded: {not_discarded} - total: {discarded_img+not_discarded}')
            print(f"items: {len(arr_weigths)}")
            print(f"mean(black, white): {np.mean(arr_weigths, axis=0)}\n")
    
    @staticmethod
    def move_ignored_items(data_path):
        PreProcessingData.move_all_ignored_folders_to_test(data_path)
    
    @staticmethod
    def move_all_ignored_folders_to_test(data_path):
        PreProcessingData.move_ignored_folder_to_test(data_path, "train", "images")
        PreProcessingData.move_ignored_folder_to_test(data_path, "train", "masks")

        PreProcessingData.move_ignored_folder_to_test(data_path, "val", "images")
        PreProcessingData.move_ignored_folder_to_test(data_path, "val", "masks")

        PreProcessingData.move_ignored_folder_to_test(data_path, "aug", "images")
        PreProcessingData.move_ignored_folder_to_test(data_path, "aug", "masks")

    @staticmethod
    def move_ignored_folder_to_test(data_path, stage, folder):
        img_path = data_path.split("/")
        image_folder = img_path
        
        if(stage!='aug'):
            base_stage_path = os.path.join("/".join(image_folder)) + f"{stage}_splits/ignore/{folder}/"
        else:
            base_stage_path = os.path.join("/".join(image_folder)) + f"{stage}/ignore/{folder}/"

        dest_stage_path = os.path.join("/".join(image_folder)) + f"test_splits/{folder}/"

        if(os.path.exists(base_stage_path) and os.path.exists(dest_stage_path)):
            i = 0
            for each_file in Path(base_stage_path).glob('*.JPG'):
                shutil.move(str(each_file), dest_stage_path) 
                i = i + 1
            
            print(f'{i} images moved from {base_stage_path} to {dest_stage_path}')
        else:
            print(f"The directory does not exists: {base_stage_path} or {dest_stage_path}")
    
    @staticmethod
    def should_I_discard(im, threshold):    
        w,h = im.size
        colors = im.getcolors(w*h)
        colordict = { x[1]:x[0] for x in colors }

        # get the amount of black pixels in image in RGB black is 0,0,0
        blackpx = colordict.get((0,0,0))

        # get the amount of white pixels in image in RGB white is 255,255,255
        whitepx = colordict.get((255,255,255))  

        if(not blackpx):
            blackpx = 0

        if(not whitepx):
            whitepx = 0    

        # percentage
        w,h = im.size
        totalpx = w*h
        blackpercent = (blackpx/totalpx)*100
        whitepercent = (whitepx/totalpx)*100

        discard = (whitepercent < threshold)

        return discard, (blackpercent, whitepercent)

    @staticmethod
    def get_name(file_name, frame_num, discard_folder):
        x = file_name.split("/")
        size = len(x)
        
        if(discard_folder):
            x[size - 3] = x[size - 3] + "_splits/ignore"
        else:
            x[size - 3] = x[size - 3] + "_splits/"

        name_ext = f"{x[size-1]}".split(".")
        name = name_ext[0] + "_part_{:02}.".format(frame_num) + name_ext[1]
        x[size-1] = name
        x = os.path.join("/".join(x))

        return x
        
    @staticmethod
    def create_split_dirs(data_path, force_delete):
        img_path = data_path.split("/")
        size = len(img_path)
        img_path[size - 2] = img_path[size - 2] + "_splits"
        x = os.path.join("/".join(img_path))
        img_path_masks = img_path
        img_path_masks[size - 1] = "masks"
        y = os.path.join("/".join(img_path_masks))

        if os.path.exists(x) or os.path.exists(y):
            if(not force_delete):
                return True
            else:
                shutil.rmtree(x, ignore_errors=False, onerror=None)
                shutil.rmtree(y, ignore_errors=False, onerror=None)
        
        os.makedirs(x + "/")
        os.makedirs(y + "/")

        PreProcessingData.create_split_dir_to_ignore(data_path.split("/"), size)
        PreProcessingData.create_split_dir_to_ignore(data_path.split("/"), size, True)

        return False

    @staticmethod
    def create_split_dir_to_ignore(ignore_path, size, is_mask = False):
        ignore_path[size - 2] = ignore_path[size - 2] + "_splits"
        
        if(not is_mask):
            ignore_path[size - 1] = "ignore/images"
        else: 
            ignore_path[size - 1] = "ignore/masks"

        ignore_path = os.path.join("/".join(ignore_path))

        if not os.path.exists(ignore_path):
            os.makedirs(ignore_path + "/")