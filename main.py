from model import *
from data import *
from pathlib import Path
from datetime import datetime

import sys 

# pip install scikit-image
# pip install keras
# pip install tensorflow


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# datetime object containing current date and time
now = datetime.now()  
dt_string = now.strftime("%Y.%m.%d_%H%M%S")


if (False):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    pathImg = 'img/' + dt_string 
    Path(pathImg).mkdir(parents=True, exist_ok=True) 
    
    myGene = trainGenerator(2, 'data/train', 'image', 'label', data_gen_args, target_size=(256, 256), image_color_mode = "rgb", save_to_dir=pathImg)

    model = unet(pretrained_weights=None, input_size=(256, 256, 3))
    model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=30, epochs=5, callbacks=[model_checkpoint])  
    
else:    
    model = unet(pretrained_weights=None, input_size=(256, 256, 3))
    model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)


    path = "data/test/predict/" + dt_string 
    Path(path).mkdir(parents=True, exist_ok=True) 

    testGene = testGenerator("data/test", flag_multi_class=True, target_size=(256, 256, 3), as_gray=False)
    results = model.predict_generator(testGene, 14, verbose=1)
    saveResult(path, results)


