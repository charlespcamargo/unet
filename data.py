from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.util import img_as_float, img_as_ubyte
from pathlib import Path

from sklearn.utils import class_weight
from PIL import Image
from tensorflow.python.keras.preprocessing.image import DirectoryIterator


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
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # get_class_weights(mask)

    return (img, mask)


def get_class_weights(mask):
    (unique, counts) = np.unique(mask, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    x = frequencies[0][1]
    y = frequencies[1][1]
    total = x + y

    x_weights = x / total
    y_weights = y / total

    return (x_weights, y_weights)


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
    num_class=1,
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
    )

    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def test_generator(
    path,
    ext=".JPG",
    num_image=30,
    target_size=(256, 256),
    flag_multi_class=False,
    as_gray=True,
):
    imgs = glob.glob(path + "/*" + ext)
    for item in imgs:
        # os.path.join(test_path,"%d.jpg"%i)
        img = io.imread(item, as_gray=as_gray)
        img = img / 255.0
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def gene_data_npy(
    data_path,
    flag_multi_class=False,
    num_class=1,
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
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
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


def label_visualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(save_path, npyfile, imgs, flag_multi_class=False, num_class=1):

    Path(save_path).mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(npyfile):
        # if flag_multi_class:
        #    img = label_visualize(num_class,COLOR_DICT,item)
        # else:
        img = item[:, :, 0]

        img[img > 0.5] = 1
        img[img <= 0.5] = 0

        io.imsave(os.path.join(save_path, imgs[i] + "_predict.png"), img_as_ubyte(img))


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

        if(index % 50 == 0):
            print(f"{index}/{len(image_name_arr)}")

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
        print(f'{data_folder} - discarded_img: {discarded_img} - not_discarded: {not_discarded} - total: {discarded_img+not_discarded}')
        print(f"items: {len(arr_weigths)}")
        print(f"\nmean(black, white): {np.mean(arr_weigths, axis=0)}\n")

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

def move_all_ignored_folders_to_test(data_path):
    move_ignored_folder_to_test(data_path, "train", "images")
    move_ignored_folder_to_test(data_path, "train", "masks")

    move_ignored_folder_to_test(data_path, "test", "images")
    move_ignored_folder_to_test(data_path, "test", "masks")

    move_ignored_folder_to_test(data_path, "val", "images")
    move_ignored_folder_to_test(data_path, "val", "masks")

def move_ignored_folder_to_test(data_path, stage, folder):
    img_path = data_path.split("/")
    image_folder = img_path
    
    base_stage_path = os.path.join("/".join(image_folder)) + f"{stage}_splits/ignore/{folder}/"
    dest_stage_path = os.path.join("/".join(image_folder)) + f"val_splits/{folder}/"

    if(os.path.exists(base_stage_path) and os.path.exists(dest_stage_path)):
        for each_file in Path(base_stage_path).glob('*.JPG'):
            shutil.move(str(each_file), dest_stage_path) 
    else:
        print(f"The directory does not exists: {base_stage_path} or {dest_stage_path}")
    

def create_split_dirs(data_path):
    img_path = data_path.split("/")
    size = len(img_path)
    img_path[size - 2] = img_path[size - 2] + "_splits"
    x = os.path.join("/".join(img_path))

    if os.path.exists(x):
        shutil.rmtree(x, ignore_errors=False, onerror=None)

    os.makedirs(x + "/")

    img_path[size - 1] = "masks"
    y = os.path.join("/".join(img_path))
    
    if os.path.exists(y):
        shutil.rmtree(x, ignore_errors=False, onerror=None)

    os.makedirs(y + "/")

    create_split_dir_to_ignore(data_path.split("/"), size)
    create_split_dir_to_ignore(data_path.split("/"), size, True)

def create_split_dir_to_ignore(ignore_path, size, is_mask = False):
    ignore_path[size - 2] = ignore_path[size - 2] + "_splits"
    
    if(not is_mask):
        ignore_path[size - 1] = "ignore/images"
    else: 
        ignore_path[size - 1] = "ignore/masks"

    ignore_path = os.path.join("/".join(ignore_path))

    if not os.path.exists(ignore_path):
        os.makedirs(ignore_path + "/")

def should_I_discard(im):    
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
    whitepercent = (whitepx/totalpx)*100
    blackpercent = (blackpx/totalpx)*100

    discard = (whitepercent < 25)

    return discard, (blackpercent, whitepercent)