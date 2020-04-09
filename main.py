from model import *
from data import *
from pathlib import Path
from datetime import datetime
import sys
import argparse
import os
import os.path 
import glob 

# pip install scikit-image
# pip install keras
# pip install tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def getFolderName(basePath):
    now = datetime.now()
    path = now.strftime("%Y.%m.%d_%H%M%S")
    Path(basePath + path).mkdir(parents=True, exist_ok=True)
    return basePath + path


def getFilesCount(path, ext = '.JPG'):    
    parts = len(path.split('/'))
    imgs = glob.glob('data/test/*' + ext)

    for i, item in enumerate(imgs):
        imgs[i] = imgs[i].split('/')[parts]
        
    return len(imgs), imgs


def main():
    parser = argparse.ArgumentParser(description='Informe os parametros:')
    parser.add_argument("--t", default=-1, type=int,
                        help="Informe o tipo '--t 0' para treino ou '--t 1' para teste'")
    parser.add_argument("--g", default=0, type=int,
                        help="Gerar arquivos de '--g 0' para nao gerar arquivos ou '--g 1' para gerar")
    parser.add_argument("--q", default=0, type=int,
                        help="Quantidade de arquivos para teste '--q 0' para nao gerar arquivos ou '--q 1' para gerar")
    args = parser.parse_args()

    if (args.t != 0 and args.t != 1):
        print("Tipo invalido! Informe o tipo corretamente: '--t 0' para treino ou '--t 1' para teste'")
        sys.exit()

    if (args.g != 0 and args.g != 1):
        print("Parametro para a geracao de arquivos invalido! Informe corretamene: '--g 0' para nao gerar arquivos ou '--g 1' para gerar")
        sys.exit()

    if (args.t == 0):
        data_gen_args = dict(rotation_range=0.2,
                             zoom_range=0.15,  
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             shear_range=0.15,                              
                             horizontal_flip=True, 
                             fill_mode='wrap')

        save_to_dir = None
        if (args.g != 0):
            save_to_dir = getFolderName('data/train/aug/')

        myGene = trainGenerator(1, 'data/train', 'image', 'label', 
                                data_gen_args, 
                                target_size=(1280, 1792), 
                                image_color_mode="rgb", 
                                save_to_dir=save_to_dir)

        model = unet(pretrained_weights=None, input_size=(1280, 1792, 3))
        model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(myGene, steps_per_epoch=1, epochs=1, callbacks=[model_checkpoint])

    elif (args.t == 1):
        model = unet(pretrained_weights='unet_hdf5', input_size=(1280, 1792, 3))
        model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)

        testGene = testGenerator('data/test', flag_multi_class=True, target_size=(1280, 1792, 3), as_gray=False)
        qtd, imgs = getFilesCount('data/test')

        if(qtd > 0):
            results = model.predict_generator(testGene, qtd, verbose=1)
            saveResult('data/predict/', npyfile=results, imgs=imgs, flag_multi_class=False)
        else:
            print("nenhum arquivo encontrado")

if __name__ == "__main__":
    main() 