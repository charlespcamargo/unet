from model import *
from data import *
from pathlib import Path
from datetime import datetime
import sys
import argparse
import os
import os.path 
import glob 
import traceback 

# pip install scikit-image
# pip install keras
# pip install tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

model = None
batch_size  = 16
target_size = (640, 896)      #(1280, 1792) #
input_shape = (640, 896, 3)   #(1280, 1792, 3) #
steps_per_epoch = 1
epochs = 1


base_folder = 'data/'

train_folder = base_folder + 'train/'
augmentation_folder = train_folder + 'aug/'

test_folder = base_folder + 'test/'

image_folder = 'image'
label_folder = 'label'

def main():
    args = arguments()

    if (args.t == -1):
        showParameters()
    
    if (args.t == 0):
        train(args)

    elif (args.t == 1):
        test(args)

    elif (args.t == 2): 
        showSummary(args)

def showParameters():
    print('batch_size: ', batch_size)
    print('target_size: ', target_size)
    print('input_shape: ', input_shape)
    print('steps_per_epoch: ', steps_per_epoch)
    print('epochs: ', epochs)
    print('base_folder: ', base_folder)
    print('train_folder: ', train_folder)
    print('augmentation_folder: ', augmentation_folder)
    print('test_folder: ', test_folder)
    print('image_folder: ', image_folder)
    print('label_folder: ', label_folder)

def getFolderName(basePath):
    now = datetime.now()
    path = now.strftime("%Y.%m.%d_%H%M%S")
    Path(basePath + path).mkdir(parents=True, exist_ok=True)
    return basePath + path

def getFilesCount(path, ext = '.JPG'):
    parts = len(path.split('/')) - 1
    imgs = glob.glob(path + '/*' + ext)
    
    for i, item in enumerate(imgs):
        imgs[i] = imgs[i].split('/')[parts]
        
    return len(imgs), imgs

def arguments():
    # show options, get arguments and validate    
    parser = argparse.ArgumentParser(description='Informe os parametros:')
    parser.add_argument("--t", default=-1, type=int,
                        help="Informe o tipo  '--t -1' parametros, '--t 0' treino, '--t 1' teste', '--t 2' sumario', '--t 3' avaliacao''")
    parser.add_argument("--g", default=0, type=int,
                        help="Gerar arquivos de '--g 0' para nao gerar arquivos ou '--g 1' para gerar")
    parser.add_argument("--q", default=0, type=int,
                        help="Quantidade de arquivos para teste '--q 0' para nao gerar arquivos ou '--q 1' para gerar")
    parser.add_argument("--n", default=None, type=str,
                        help="Informe o nome do arquivo de pesos para ler o sumario")
    args = parser.parse_args()

    if (args.t != -1 and args.t != 0 and args.t != 1 and args.t != 2):
        print("Tipo invalido! Informe o tipo corretamente: --t -1' parametros, '--t 0' para treino, '--t 1' para teste', '--t 2' para exibir o sumario'")
        sys.exit()

    if (args.g != 0 and args.g != 1):
        print("Parametro para a geracao de arquivos invalido! Informe corretamente: '--g 0' para nao gerar arquivos ou '--g 1' para gerar")
        sys.exit()
    
    if ((args.t == 2) and not args.n):
        print("Parametro invalido! Informe corretamente: '--n [file_name]'")
        sys.exit()

    return args

def train(args):
    data_gen_args = dict(rotation_range=0.2,
                             zoom_range=0.15,  
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             shear_range=0.15,                              
                             horizontal_flip=True, 
                             fill_mode='wrap')

    save_to_dir = None
    if (args.g != 0):
        save_to_dir = getFolderName(augmentation_folder)

    myGene = trainGenerator(batch_size, 
                            train_folder, 
                            image_folder,
                            label_folder, 
                            data_gen_args, 
                            target_size=target_size, 
                            image_color_mode="rgb", 
                            save_to_dir=save_to_dir)


    
    path = datetime.now().strftime("%Y%m%d_%H%M")

    # define TensorBoard directory
    tb_dir = f'.logs/{path}'
    
    # define TensorBoard callback
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_dir, write_graph=True, update_freq=100)


    model = unet(pretrained_weights=None, input_size=input_shape)
    model_checkpoint = ModelCheckpoint('unet_hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[tb_cb])

    print('acabou! {datetime.now().strftime("%Y%m%d_%H%M")}')

def test(args):

    if(not args.n):
        args.n = 'unet_hdf5'

    model = unet(pretrained_weights=args.n, input_size=input_shape)
    model_checkpoint = ModelCheckpoint(args.n, monitor='loss', verbose=1, save_best_only=True)

    testGene = testGenerator(test_folder + image_folder + '/', flag_multi_class=True, target_size=input_shape, as_gray=False)
    qtd, imgs = getFilesCount(test_folder + image_folder + '/')

    print("Images do predict: {0} - {1}".format(qtd, len(imgs)))

    if(qtd > 0):
        results = model.predict_generator(testGene, qtd, verbose=1)
        saveResult(test_folder + label_folder + '/', npyfile=results, imgs=imgs, flag_multi_class=False)

        try:
            #loss, acc = model.evaluate(x=testGene, y=results, verbose=1)
            
            #print("Restored model, accuracy: {:5.2f}%".format(100*acc))
            #print("Restored model, loss: {:5.2f}%".format(100*loss))        
            print("Not yet!")        
        except Exception as e:
            print("type error: " + str(e))
            print(traceback.format_exc())
            pass

    else:
        print("nenhum arquivo encontrado")

def showSummary(args):
    model = unet(pretrained_weights=args.n, input_size=input_shape)        
    model.build(input_shape)
    model.summary()

if __name__ == "__main__":
    main() 