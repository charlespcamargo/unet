from data import *
from PIL import Image

class PreProcessingData():

    @staticmethod
    def crop_image(data_path, data_folder, image_folder="images", mask_folder="masks", w=400, h=320, validate_class_to_discard = False):
        image_path = os.path.join(data_path, data_folder, image_folder)
        mask_path = os.path.join(data_path, data_folder, image_folder)
        image_name_arr = glob.glob(os.path.join(image_path, "*.JPG"))
        discarded_img = 0
        not_discarded = 0

        arr_weigths = []
        discard_folder = False

        create_split_dirs(image_path)

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
                    croped = img.crop((col_i, row_i, col_i + w, row_i + h))
                    croped_mask = mask.crop((col_i, row_i, col_i + w, row_i + h))
                    
                    if(validate_class_to_discard):
                        discard_folder, (blackpercent, whitepercent)  = should_I_discard(croped_mask)
                    
                        if(not discard_folder):
                            arr_weigths.append((blackpercent, whitepercent))

                        if(discard_folder):
                            discarded_img = discarded_img + 1
                        else:
                            not_discarded = not_discarded + 1               

                    croped.save(get_name(file_name, frame_num, discard_folder))                
                    croped_mask.save(get_name(mask_name, frame_num, discard_folder)) 
                    
                    frame_num += 1

        if(validate_class_to_discard):
            print(f'\n{data_folder} - discarded_img: {discarded_img} - not_discarded: {not_discarded} - total: {discarded_img+not_discarded}')
            print(f"items: {len(arr_weigths)}")
            print(f"mean(black, white): {np.mean(arr_weigths, axis=0)}\n")

    @staticmethod
    def crop_images_in_tiles(data_path, train_folder, val_folder, test_folder, image_folder, mask_folder, w, h):
        if(train_folder):
            PreProcessingData.crop_image(data_path, train_folder, image_folder, mask_folder , w, h, True)
        if(val_folder):
            PreProcessingData.crop_image(data_path, val_folder, image_folder, mask_folder , w, h)
        if(test_folder):
            PreProcessingData.crop_image(data_path, test_folder, image_folder, mask_folder , w, h)

        PreProcessingData.move_ignored_items(data_path)

    @staticmethod
    def move_ignored_items(data_path):
        PreProcessingData.move_all_ignored_folders_to_test(data_path)
    
    @staticmethod
    def move_all_ignored_folders_to_test(data_path):
        PreProcessingData.move_ignored_folder_to_test(data_path, "train", "images")
        PreProcessingData.move_ignored_folder_to_test(data_path, "train", "masks")

        PreProcessingData.move_ignored_folder_to_test(data_path, "test", "images")
        PreProcessingData.move_ignored_folder_to_test(data_path, "test", "masks")

        PreProcessingData.move_ignored_folder_to_test(data_path, "val", "images")
        PreProcessingData.move_ignored_folder_to_test(data_path, "val", "masks")

    @staticmethod
    def move_ignored_folder_to_test(data_path, stage, folder):
        img_path = data_path.split("/")
        image_folder = img_path
        
        base_stage_path = os.path.join("/".join(image_folder)) + f"{stage}_splits/ignore/{folder}/"
        dest_stage_path = os.path.join("/".join(image_folder)) + f"test_splits/{folder}/"

        if(os.path.exists(base_stage_path) and os.path.exists(dest_stage_path)):
            i = 0
            for each_file in Path(base_stage_path).glob('*.JPG'):
                shutil.move(str(each_file), dest_stage_path) 
                i = i + 1
            
            print(f'{i} images moved from {base_stage_path} to {dest_stage_path}')
        else:
            print(f"The directory does not exists: {base_stage_path} or {dest_stage_path}")
    