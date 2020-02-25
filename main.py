from model import *
from data import *
from pathlib import Path
from datetime import datetime
import sys
import argparse
import os
import os.path 

# pip install scikit-image
# pip install keras
# pip install tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def getFolderName(basePath):
    now = datetime.now()
    path = now.strftime("%Y.%m.%d_%H%M%S")
    Path(basePath + path).mkdir(parents=True, exist_ok=True)
    return basePath + path


def getFilesCount(path):
    path, dirs, files = next(os.walk(path))
    return len(files)


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

        myGene = trainGenerator(2, 'data/train', 'image', 'label', 
                                data_gen_args, 
                                target_size=(512, 512), 
                                image_color_mode="rgb", 
                                save_to_dir=save_to_dir)

        model = unet(pretrained_weights=None, input_size=(512, 512, 3))
        model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(myGene, steps_per_epoch=20, epochs=50, callbacks=[model_checkpoint])

    elif (args.t == 1):
        model = unet(pretrained_weights='unet_hdf5', input_size=(512, 512, 3))
        model_checkpoint = ModelCheckpoint(
            'unet_hdf5', monitor='loss', verbose=1, save_best_only=True)

        testGene = testGenerator('data/test', flag_multi_class=True, target_size=(512, 512, 3), as_gray=False)
        qtd = getFilesCount('data/test')

        if(qtd > 0):
            results = model.predict_generator(testGene, qtd, verbose=1)
            saveResult('data/predict/', results, flag_multi_class=False)
        else:
            print("nenhum arquivo encontrado")


# test - DJI_0179.jpg
if __name__ == "__main__":
    main()

# 512 x 512 - sem gerar imagem - 5 steps - 2 epocas
# 0.69 - 0.95
# 0.39 - 0.95

# 256 x 256 - sem gerar imagem - 5 steps - 2 epocas
# 0.56 - 0.65
# 0.69 - 0.95

# 1024 x 1024 - sem gerar imagem - 5 steps - 2 epocas
# 0.00 - 0.00
# 0.00 - 0.00