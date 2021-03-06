from PIL import Image
import os
import glob
import numpy as np
from pathlib import Path
import shutil


class PreProcessingData():


    @staticmethod
    def crop_all_images_in_tiles(data_path, image_folder, mask_folder, w, h, threshold = 5, force_delete = False, validate_class_to_discard = True, move_ignored_to_test = False):
        PreProcessingData.crop_all_image(data_path, image_folder, mask_folder , w, h, threshold, force_delete, validate_class_to_discard)

        if(move_ignored_to_test):
            PreProcessingData.move_all_ignored_folders_to_test(data_path)

    @staticmethod
    def crop_all_image(data_path, image_folder="images", mask_folder="masks", w=256, h=256, threshold = 5, force_delete = False, validate_class_to_discard = False):
        image_path = os.path.join(data_path, image_folder)
        mask_path = os.path.join(data_path, image_folder)
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
                print(f"{index}/{len(image_name_arr)} - processando.")

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
                        discard_folder, (blackpercent, whitepercent)  = PreProcessingData.should_discard(croped_mask, threshold)
                    
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
            print(f'\ndiscarded_img: {discarded_img} - not_discarded: {not_discarded} - total: {discarded_img+not_discarded}')
            print(f"items: {len(arr_weigths)}")
            print(f"mean(black, white): {np.mean(arr_weigths, axis=0)}\n")

    @staticmethod
    def crop_images_in_tiles(data_path, train_folder, val_folder, test_folder, image_folder, mask_folder, w, h, threshold = 50, force_delete = False, move_ignored_to_test = False):
        if(train_folder):
            PreProcessingData.crop_image(data_path, train_folder, image_folder, mask_folder , w, h, threshold, force_delete, True)
        if(val_folder):
            PreProcessingData.crop_image(data_path, val_folder, image_folder, mask_folder , w, h, threshold, force_delete, True)
        if(test_folder):
            PreProcessingData.crop_image(data_path, test_folder, image_folder, mask_folder , w, h, threshold, force_delete, False)

        if(move_ignored_to_test):
            PreProcessingData.move_all_ignored_folders_to_test(data_path)

    @staticmethod
    def crop_image(data_path, data_folder, image_folder="images", mask_folder="masks", w=256, h=256, threshold = 20, force_delete = False, validate_class_to_discard = False):
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
    def should_discard(im, threshold):    
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
   
    @staticmethod
    def get_train_class_weights(data_path, use_splits = False):
        _splits = ''
        if(use_splits):
            _splits = '_splits'
            
        img_path = data_path.split("/")
        base_train_path = os.path.join("/".join(img_path)) + f"train{_splits}/masks/"
        
        i = 0
        arr_weigths = []
        for each_mask in Path(base_train_path).glob('*.JPG'):
            img = Image.open(each_mask)
            black = (0, 0, 0)
            white = (255, 255, 255) 

            px_black = 0
            px_white = 0
            px_other = 0

            for pixel in img.getdata():
                if pixel == black:
                    px_black += 1
                elif pixel == white:
                    px_white += 1
                else:
                    px_other += 1

            total = px_black + px_white + px_other
            x_weights = round(px_black / total, 2)
            y_weights = round(px_white / total, 2)            

            arr_weigths.append((x_weights, y_weights))
            if(i > 0 and i % 500 == 0):
                print(f'i: {i} - calc the class weights')
            
            i += 1

        if(len(arr_weigths)):
            print("Any data was found!")
            return (0, 0)

        (black, white) = np.mean(arr_weigths, axis=0)                
        print(f"\nmean(black, white): {black}, {white}\n")

        return (round(black, 2), round(white, 2))
