from PIL import Image
from exif import Image as ImageExif

import os
import glob
import numpy as np
from pathlib import Path
import shutil
import skimage.io as io


class PreProcessingData():


    @staticmethod
    def crop_all_images_in_tiles(data_path, image_folder, mask_folder, w, h, threshold = 5, force_delete = False, validate_class_to_discard = True, move_ignored_to_test = False, amount_of_classless_image_to_copy = 10):
        PreProcessingData.crop_all_image(data_path, image_folder, mask_folder , w, h, threshold, force_delete, validate_class_to_discard)

        if(move_ignored_to_test):
            PreProcessingData.move_all_ignored_folders_to_test(data_path)        
        elif(amount_of_classless_image_to_copy > 0):            
            PreProcessingData.split_dataset(data_path, 80, 10, 10)

    @staticmethod
    def split_dataset(data_path, train, val, test):
        if(train <= 0):
            print("the train need to be more than zero")
            return
        
        if(val + test <= 0):
            print("Is necessary to have val or test")
            return
        
        if(val < 0 or test < 0):
            print("val/test cannot be less than zero")
            return

        if( train + val + test > 100):
            print(f"The amount({train+val+test}) cannot be more than 100%")
            return

        img_path = list(filter(lambda x: x != "", data_path.split("/")))
        image_folder = img_path
        data_path = "/" + os.path.join("/".join(image_folder)) + f"_splits/"
        qty_files = len(os.listdir(data_path + 'images/'))

        if(qty_files <= 0):
            print("there is no file to splits")
            return

        qty_train = round(qty_files * train / 100)
        qty_val = round(qty_files * val / 100) if val > 0 else 0
        qty_test = round(qty_files * test / 100) if test > 0 else 0
        #rounding
        qty_train += qty_files-(qty_train+qty_val+qty_test)
        
        # 3% classless for all bases
        classless = 0.03
        qty_classless_train = round(qty_train * classless)
        qty_classless_val = round(qty_val * classless) if qty_val > 0 else 0
        qty_classless_test = round(qty_test * classless) if qty_test > 0 else 0

        if(not os.path.exists(data_path + 'train/')):
            os.makedirs(data_path + 'train/')
            os.makedirs(data_path + 'train/images/')
            os.makedirs(data_path + 'train/masks/')
        
        if(not os.path.exists(data_path + 'val/') and qty_val > 0):
            os.makedirs(data_path + 'val/')
            os.makedirs(data_path + 'val/images/')
            os.makedirs(data_path + 'val/masks/')

        if(not os.path.exists(data_path + 'test/') and qty_test > 0):
            os.makedirs(data_path + 'test/')
            os.makedirs(data_path + 'test/images/')
            os.makedirs(data_path + 'test/masks/')

        PreProcessingData.move_image_to_stage_folder(data_path, 'train', qty_images=qty_train, qty_classless=qty_classless_train)
        PreProcessingData.move_image_to_stage_folder(data_path, 'val', qty_images=qty_val, qty_classless=qty_classless_val)
        PreProcessingData.move_image_to_stage_folder(data_path, 'test', qty_images=qty_test, qty_classless=qty_classless_test)
        



        

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
            mask = io.imread(mask_name, as_gray=False)
                        
            #mask = Image.open(mask_name)
            # x = np.where(np.logical_and(mask > 0, mask < 255))
            # r = len(x[0]) * 3 if x and len(x) > 0 else 0
            # print(f'{r} pixels not black and white')

            mask[mask<=127.5] = 0
            mask[mask>127.5] = 255
            mask = Image.fromarray(mask)

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
            print(f'\ntotal: {discarded_img+not_discarded} - ignored: {discarded_img} - useful: {not_discarded}')
            print(f"train/val/test: {(not_discarded * 0.80)}/{(not_discarded * 0.10)}/{(not_discarded * 0.10)}\n")
            print(f"threshold: {threshold}%")
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
    def move_image_to_stage_folder(base_stage_path, stage, qty_images, qty_classless):
        
        img_path = base_stage_path.split("/")
        image_folder = img_path

        dest_stage_path = os.path.join("/".join(image_folder), stage)
        image_folder = base_stage_path + 'images/'
        mask_folder = base_stage_path + 'masks/' 

        i = 0

        for each_file in Path(image_folder).glob('*.JPG'):
            shutil.move(each_file, dest_stage_path + '/images/')
            shutil.move(mask_folder + each_file.name, dest_stage_path + '/masks/') 

            if(i >= qty_images):
                break

            i += 1
        
        print(f'{i} images moved from {base_stage_path} to {dest_stage_path}')

        i = 0

        image_folder = base_stage_path + '/ignore/images/'
        mask_folder = base_stage_path + '/ignore/masks/' 

        for each_file in Path(mask_folder).glob('*.JPG'):
            img = Image.open(each_file)
            blackpercent, whitepercent = PreProcessingData.count_black_and_white(img)

            if(whitepercent > 0):
                continue
            
            if(i >= qty_classless):
                break

            shutil.move(image_folder + each_file.name, dest_stage_path + '/images/')
            shutil.move(each_file, dest_stage_path + '/masks/') 

            i += 1 
            
        
        print(f'{i} images moved from {base_stage_path} to {dest_stage_path}')

    
    
    @staticmethod
    def should_discard(img, threshold):    
        blackpercent, whitepercent = PreProcessingData.count_black_and_white(img)
        discard = (whitepercent < threshold)

        return discard, (blackpercent, whitepercent)

    @staticmethod
    def count_black_and_white(img):
        w,h = img.size
        totalpx = w*h
        colors = img.getcolors(w*h)
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
        blackpercent = (blackpx/totalpx)*100
        whitepercent = (whitepx/totalpx)*100

        return blackpercent, whitepercent
    

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

    @staticmethod
    def copy_exif_from_image_to_mask(base_path, image_folder, mask_folder):
        image_path = os.path.join(base_path, image_folder)
        mask_path = os.path.join(base_path, image_folder)
        image_name_arr = glob.glob(os.path.join(image_path, "*.JPG"))
        
        print(f'total of files: {len(image_name_arr)}')

        for index, item in enumerate(image_name_arr):

            file_name = item
            mask_name = item.replace(image_path, mask_path).replace(image_folder, mask_folder)

            if(index % 25 == 0):
                print(f'current index: {index}')

            with open(file_name, "rb") as image_buffer:
                image = ImageExif(image_buffer)
                if not image.has_exif:
                    print(f"[{file_name}] - does not contain any EXIF information.")
                #else:
                    #PreProcessingData.show_exif_data(index, image)

            with open(mask_name, "rb") as mask_buffer:
                mask = ImageExif(mask_buffer)
                if not mask.has_exif:
                    print(f"[{mask_name}] - does not contain any EXIF information.")
                else:
                    mask.focal_length = image.focal_length
                    mask.focal_length_in_35mm_film = image.focal_length_in_35mm_film
                    mask.shutter_speed_value = image.shutter_speed_value
                    mask.gps_altitude = image.gps_altitude
                    mask.f_number = image.f_number
                    mask.compressed_bits_per_pixel = image.compressed_bits_per_pixel
                    mask.exposure_time = image.exposure_time
                    mask.datetime_original = image.datetime_original
                    mask.x_resolution = image.x_resolution
                    mask.y_resolution = image.y_resolution
                    mask.gps_latitude = image.gps_latitude
                    mask.gps_latitude_ref = image.gps_latitude_ref
                    mask.gps_longitude = image.gps_longitude
                    mask.gps_longitude_ref = image.gps_longitude_ref
                    mask.make = image.get('make', 'Unknown')
                    mask.model = image.get('model', 'Unknown')

                    mask_buffer.write(mask.get_file())


        
                    

    @staticmethod
    def show_exif_data(index, image):
        if image.has_exif:                    
            print(f"Lens make: {image.get('make', 'Unknown')}")
            print(f"Lens model: {image.get('model', 'Unknown')}")
            print(f"Lens specification: {image.get('lens_specification', 'Unknown')}")
            print(f"OS version: {image.get('software', 'Unknown')}\n")
            print(f"Date/time taken - Image {index}") 
            print(f"Focal length: {image.focal_length}")
            print(f"Focal length in 35mm film: {image.focal_length_in_35mm_film}")
            print(f"shutter_speed_value: {image.shutter_speed_value}")
            print(f"Altitude: {image.gps_altitude}")
            print(f"f_number: {image.f_number}")    
            print(f"compressed_bits_per_pixel: {image.compressed_bits_per_pixel}")    
            print(f"exposure_time: {image.exposure_time}")
            print(f"{image.datetime_original} {image.get('offset_time', '')}\n")                    
            print(f"Resolution - x:{image.x_resolution} y:{image.y_resolution}")
            print(f"Latitude: {image.gps_latitude} {image.gps_latitude_ref}")
            print(f"Longitude: {image.gps_longitude} {image.gps_longitude_ref}\n")                    
            print(f"Latitude (DMS): {PreProcessingData.format_dms_coordinates(image.gps_latitude)} {image.gps_latitude_ref}")
            print(f"Longitude (DMS): {PreProcessingData.format_dms_coordinates(image.gps_longitude)} {image.gps_longitude_ref}\n")
            print(f"Latitude (DD): {PreProcessingData.dms_coordinates_to_dd_coordinates(image.gps_latitude, image.gps_latitude_ref)}")
            print(f"Longitude (DD): {PreProcessingData.dms_coordinates_to_dd_coordinates(image.gps_longitude, image.gps_longitude_ref)}\n")


    @staticmethod
    def format_dms_coordinates(coordinates):
        return f"{coordinates[0]}° {coordinates[1]}\' {coordinates[2]}\""

    @staticmethod
    def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
        decimal_degrees = coordinates[0] + \
                        coordinates[1] / 60 + \
                        coordinates[2] / 3600
        
        if coordinates_ref == "S" or coordinates_ref == "W":
            decimal_degrees = -decimal_degrees
        
        return decimal_degrees