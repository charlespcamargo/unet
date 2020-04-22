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
# curva roc e UAC

# training vars
model = None 
batch_size  = 4
steps_per_epoch = 200
epochs = 200

# image sizes
target_size = (640, 896)      #(1280, 1792) #
input_shape = (640, 896, 3)   #(1280, 1792, 3) #    

# paths
base_folder = 'data/'
train_folder = base_folder + 'train/'
augmentation_folder = train_folder + 'aug/'
test_folder = base_folder + 'test/'
image_folder = 'image'
label_folder = 'label'

tz = pytz.timezone("Brazil/East")

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

def getFilesCount(path, ext = '.JPG'):
    parts = len(path.split('/')) - 1
    imgs = glob.glob(path + '/*' + ext)
    
    for i, item in enumerate(imgs):
        imgs[i] = imgs[i].split('/')[parts]
        
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


    start_time = datetime.now(tz=tz)
    path = start_time.strftime("%Y%m%d_%H%M")

    # define TensorBoard directory and TensorBoard callback
    tb_dir = f'.logs/{path}'
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_dir, write_graph=True, update_freq=100)

    try:
        showExecutionTime(start_time, originalMsg='Starting now...', writeInFile=True)

        model = unet(pretrained_weights=None, input_size=input_shape)
        model_checkpoint = ModelCheckpoint(f'train_weights/{path}_unet.hdf5', monitor='loss', verbose=0, save_best_only=True)
        model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint, tb_cb])

        showExecutionTime(start_time, writeInFile=True)

    except Exception as e:
        showExecutionTime(start_time, success=False, writeInFile=True)

        error_Msg = "\ntype error: " + str(e) + ' \ntraceback: ' + traceback.format_exc()
        showExecutionTime(start_time, success=False, originalMsg=error_Msg, writeInFile=True)
        pass

def test(args):

    if(not args.n):
        args.n = 'train_weights/unet.hdf5'
    else:
        args.n = 'train_weights/' + args.n

    model = unet(pretrained_weights=args.n, input_size=input_shape)
    model_checkpoint = ModelCheckpoint(args.n, monitor='loss', verbose=1, save_best_only=True)

    testGene = testGenerator(test_folder + image_folder + '/', flag_multi_class=True, target_size=input_shape, as_gray=False)
    qtd, imgs = getFilesCount(test_folder + image_folder + '/')
   
    
    start_time = datetime.now(tz=tz)
    path = start_time.strftime("%Y%m%d_%H%M")

    # define TensorBoard directory and TensorBoard callback
    tb_dir = f'.logs/{path}'
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_dir, write_graph=True, update_freq=1)

    if(qtd > 0):

        try:
            showExecutionTime(start_time, originalMsg='Starting now...', writeInFile=True)

            results = model.predict_generator(testGene, qtd, verbose=1, callbacks=[tb_cb])
            
            createDirectory(path + test_folder + label_folder + '/')
            saveResult(test_folder + label_folder + '/', npyfile=results, imgs=imgs, flag_multi_class=True, target_size=input_shape, as_gray=False)

            showExecutionTime(start_time, originalMsg='Ending now...', writeInFile=True)    
            print("Not yet!")        


            loss, acc = model.evaluate(x=testGene, targets=results, verbose=1)            
            #print("Restored model, accuracy: {:5.2f}%".format(100*acc))
            #print("Restored model, loss: {:5.2f}%".format(100*loss))    

        except Exception as e:
            showExecutionTime(start_time, success=False, writeInFile=True)
            error_Msg = "\ntype error: " + str(e) + ' \ntraceback: ' + traceback.format_exc()
            showExecutionTime(start_time, success=False, originalMsg=error_Msg, writeInFile=True)
            pass

    else:
        print("nenhum arquivo encontrado")

def showSummary(args):
    model = unet(pretrained_weights=args.n, input_size=input_shape)        
    model.build(input_shape)
    model.summary()

def showExecutionTime(start_time, success = True, originalMsg = '', writeInFile = False):
    end_time = datetime.now(tz=tz)
    elapsed = end_time - start_time
    
    msg = originalMsg + f'\n==================================================' \
                        f'\nExecution checkpoint! ' \
                        f'\nWith success: {success} ' \
                        f'\nProcess Started: {start_time.strftime("%Y/%m/%d %H:%M")}' \
                        f'\nProcess Now: {end_time.strftime("%Y/%m/%d %H:%M")}' \
                        f'\nProcess Time elapsed: {elapsed}' \
                        f'\n==================================================' \
                        f'\n\n\n' \

    if(writeInFile):
        path = f'.logs/{start_time.strftime("%Y%m%d_%H%M")}/'
        print(f'\nlog path: {path}execution_log.txt\n\n{msg}\n')
        writeTextFile(path, 'execution_log.txt', msg)

def writeTextFile(path, file_name, text):
    createDirectory(path)
        
    text_file = open(path + file_name, "a+")
    n = text_file.write(text)
    text_file.close()

def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)



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