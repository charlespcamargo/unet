from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(2, 'data/train', 'image', 'label',
                        data_gen_args, target_size=(1024, 1024))

model = unet(pretrained_weights=None, input_size=(1024, 1024, 1))
model_checkpoint = ModelCheckpoint(
    'unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=100, epochs=5, callbacks=[model_checkpoint])

testGene = testGenerator("data/test", target_size=(1024, 1024))
results = model.predict_generator(testGene, 14, verbose=1)
saveResult("data/test/predict/", results)
