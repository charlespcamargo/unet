from model import *
from data import *
from pathlib import Path
import sys
import argparse
import os
import os.path 
import glob 
import traceback 

from datetime import datetime
from _datetime import timezone
import pytz 
import errno 

# curva roc e UAC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.callbacks.callbacks import Callback  
from kerasCustom.CustomCallback import CustomCallback
from kerasCustom.LossAndErrorPrintingCallback import LossAndErrorPrintingCallback

# curva roc e UAC

# training vars
model = None 
batch_size  = 4
steps_per_epoch = 50
epochs = 300

# image sizes
target_size = (640, 896)      #(1280, 1792) #
input_shape = (640, 896, 3)   #(1280, 1792, 3) #    

# paths
base_folder = 'data/'
train_folder = base_folder + 'train/'
augmentation_folder = train_folder + 'aug/'
test_folder = base_folder + 'test/' # 'temp_folder/'
image_folder = 'image'
label_folder = 'label'

tz = pytz.timezone("Brazil/East")
start_time = datetime.now(tz=tz)
path = None

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

    elif (args.t == 3): 
        showSummary(args)

    elif (args.t == 4):
        get_fbetaScore(args.b, args.p, args.r)

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

def getFilesCount(path, ext = '.JPG', flag_multi_class = True, target_size = (256,256), as_gray = False):
    parts = len(path.split('/'))
    imgs = glob.glob(path + '/*' + ext)
    
    for i, item in enumerate(imgs):
        imgs[i] = imgs[i].split('/')[parts]
        
    # imgs = glob.glob(path + '/*' + ext)     

    # for item in enumerate(imgs):
    #     img = io.imread(item[1], as_gray = as_gray)
    #     img = img / 255.
    #     img = trans.resize(img, target_size)
    #     img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    #     img = np.reshape(img,(1,)+img.shape)               
    #     yield img


        
    return len(imgs), imgs

def arguments():
    # show options, get arguments and validate    
    parser = argparse.ArgumentParser(description='Informe os parametros:')
    parser.add_argument("--t", default=None, type=int,
                        help="Informe o tipo  '--t -1' parametros, '--t 0' treino, '--t 1' teste, '--t 2' sumario', '--t 3' avaliacao, '--t 4' f-beta-score")
    parser.add_argument("--g", default=0, type=int,
                        help="Gerar arquivos de '--g 0' para nao gerar arquivos ou '--g 1' para gerar")
    parser.add_argument("--q", default=0, type=int,
                        help="Quantidade de arquivos para teste '--q 0' para nao gerar arquivos ou '--q 1' para gerar")
    parser.add_argument("--n", default=None, type=str,
                        help="Informe o nome do arquivo de pesos para executar o teste ou ler o sumario")
    parser.add_argument("--b", default=None, type=float,
                        help="Informe o beta para calcular o f-beta score")
    parser.add_argument("--p", default=None, type=float,
                        help="Informe o precision para calcular o f-beta score")
    parser.add_argument("--r", default=None, type=float,
                        help="Informe o recall para calcular o f-beta score")

    args = parser.parse_args()

    if (args.t != -1 and args.t != 0 and args.t != 1 and args.t != 2 and args.t != 4):
        print("Tipo invalido! Informe o tipo corretamente: --t -1' parametros, '--t 0' para treino, '--t 1' para teste', '--t 2' para exibir o sumario, '--t 4' f-beta-score")
        sys.exit()

    if (args.g != 0 and args.g != 1):
        print("Parametro para a geracao de arquivos invalido! Informe corretamente: '--g 0' para nao gerar arquivos ou '--g 1' para gerar")
        sys.exit()
    
    if ((args.t == 2) and not args.n):
        print("Parametro invalido! Informe corretamente: '--n [file_name]'")
        sys.exit()

    if ((args.t == 4) and (not args.b or not args.p or not args.r) ):
        print("Parametro invalido! Informe corretamente: '--b [beta], --p [precision], --r [recall]'")
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

    # define TensorBoard directory and TensorBoard callback
    tb_cb = createTensorBoardCallBack()

    try: 
        showExecutionTime(originalMsg='Starting now...', writeInFile=True)

        model = unet(pretrained_weights=None, input_size=input_shape)
        earlystopper = Callback(patience=5, verbose=1)
        model_checkpoint = ModelCheckpoint(f'train_weights/{path}_unet.hdf5', monitor='loss', verbose=0, save_best_only=True)
        model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[earlystopper, model_checkpoint, tb_cb])

        showExecutionTime(writeInFile=True)

    except Exception as e:
        showExecutionTime(success=False, writeInFile=True)
        error_Msg = "\ntype error: " + str(e) + ' \ntraceback: ' + traceback.format_exc()
        showExecutionTime(success=False, originalMsg=error_Msg, writeInFile=True)
        pass

def test(args):

    if(not args.n):
        args.n = 'train_weights/20200420_0817_unet-100-100-loss0_431_acc0_9837.hdf5'
    else:
        args.n = 'train_weights/' + args.n

    qtd, imgs = getFilesCount(test_folder + image_folder, target_size=target_size)

    if(qtd > 0):
        try: 
            showExecutionTime(originalMsg='Starting now...', writeInFile=True)
            
            tb_cb = createTensorBoardCallBack()
            testGene = testGenerator(test_folder + image_folder + '/', flag_multi_class=True, target_size=input_shape, as_gray=False)
            
            model = unet(pretrained_weights=args.n, input_size=input_shape) 
            results = model.predict_generator(generator=testGene, steps=qtd, callbacks=[tb_cb], verbose=1)
            
            saveResult(save_path=test_folder + '/results', npyfile=results, imgs=imgs) 



            #labelGene = testGenerator(test_folder + label_folder + '/', flag_multi_class=True, target_size=input_shape, as_gray=False)

            # myGene = trainGenerator(batch_size=batch_size, 
            #                         train_path=test_folder,  
            #                         image_folder=image_folder, 
            #                         mask_folder=label_folder,
            #                         aug_dict=None, 
            #                         target_size=target_size, 
            #                         image_color_mode="rgb")
            
            # res = model.evaluate(x=results, verbose=0, callbacks=[tb_cb])
            # res = model.predict(x=myGene, batch_size=batch_size, callbacks=[CustomCallback()])

            showExecutionTime(writeInFile=True)

        except Exception as e:
            showExecutionTime(success=False, writeInFile=True)
            error_Msg = "\ntype error: " + str(e) + ' \ntraceback: ' + traceback.format_exc()
            print(error_Msg)
            showExecutionTime(success=False, originalMsg=error_Msg, writeInFile=True)
            pass

    else:
        print("nenhum arquivo encontrado")

def showSummary(args):
    model = unet(pretrained_weights=args.n, input_size=input_shape)        
    model.build(input_shape)
    model.summary()

def showExecutionTime(success = True, originalMsg = '', writeInFile = False):
    end_time = datetime.now(tz=tz)
    elapsed = end_time - start_time
    
    msg = originalMsg + f'\n==================================================' \
                        f'\nExecution checkpoint! Success: {success} ' \
                        f'\nProcess Started: {start_time.strftime("%Y/%m/%d %H:%M")}' \
                        f'\nProcess Now: {end_time.strftime("%Y/%m/%d %H:%M")}' \
                        f'\nProcess Time elapsed: {elapsed}' \
                        f'\n==================================================' \
                        f'\n\n\n' \

    if(writeInFile):
        basePath = f'.logs/{start_time.strftime("%Y%m%d")}/'
        path = f'{start_time.strftime("%Y%m%d_%H%M")}/'
        writeTextFile(basePath, path, 'execution_log.txt', msg)

def writeTextFile(basePath, path, file_name, text):
    createDirectory(basePath, path)
        
    text_file = open(basePath + path + file_name, "a+")
    n = text_file.write(text)
    text_file.close()

def createDirectory(basePath, path):
    if (basePath and not os.path.exists(basePath)):
        os.makedirs(basePath)
    
    if (basePath + path and not os.path.exists(basePath + '/' + path)):
        os.makedirs(basePath + '/' + path)

def createTensorBoardCallBack():
    basePath = f'.logs/{start_time.strftime("%Y%m%d")}'
    path = start_time.strftime("%Y%m%d_%H%M")
    tb_dir = f'{basePath}/{path}/'
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_dir, write_graph=True, update_freq=1)
    
    createDirectory(basePath, path)

    return tb_cb


def get_recall(tp, fn):
    recall = tp / (tp + fn)
    print(f'The recall value is: {recall}')
    return recall

def get_precision(tp, tn):
    precision = tp / (tp + tp)
    print(f'The precision value is: {precision}')
    return precision

def get_f1_measure():
    return 0

def get_fscore(recall, precision):
    f_score = (2 * recall * precision) / (recall + precision)
    print(f'The f_score value is: {f_score}')
    return f_score

def get_fbetaScore(beta, precision, recall):

    # F0.5-Measure  (beta=0.5): More weight on precision, less weight on recall
    # F1-Measure    (beta=1.0): Balance the weight on precision and recall
    # F2-Measure    (beta=2.0): Less weight on precision, more weight on recall 
    f_beta_score = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
    print(f'The f_beta_score value is: {f_beta_score}')
    return f_beta_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def get_auc(model, testX, testY):
    probs = model.predict_proba(testX)
    probs = probs[:, 1]
    
    auc = roc_auc_score(testY, probs)
    print('AUC: %.2f' % auc)

    fpr, tpr, thresholds = roc_curve(testY, probs)
    plot_roc_curve(fpr, tpr)

    return auc

#Jaccard Index
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)  
  return iou

  


if __name__ == "__main__":     
    main()  