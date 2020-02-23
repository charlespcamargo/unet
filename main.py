from model import *
from data import *
from pathlib import Path
from datetime import datetime

import sys 


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if (True):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    
    myGene = trainGenerator(1, 'data/train', 'image', 'label', data_gen_args, target_size=(1024, 1024))

    model = unet(pretrained_weights=None, input_size=(256, 256, 1))
    model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=1, epochs=1, callbacks=[model_checkpoint])  
    
else:    
    model = unet(pretrained_weights='unet_hdf5', input_size=(1024, 1024, 1))
    model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
    
    # datetime object containing current date and time
    now = datetime.now()  
    dt_string = now.strftime("%Y.%m.%d_%H%M%S")
    path = "data/test/predict/" + dt_string

    Path(path).mkdir(parents=True, exist_ok=True) 

    testGene = testGenerator("data/test", target_size=(256, 256))
    results = model.predict_generator(testGene, 13, verbose=1)
    saveResult(path, results)


