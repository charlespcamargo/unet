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
import pytz
import errno

# curva roc e UAC
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import tensorflow as tf

# import tensorflow.keras
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K


class UnetHelper:
    # training vars
    model = None
    batch_size = 1
    steps_per_epoch = 50
    epochs = 15

    # image sizes
    target_size = (400, 320)  # (1280, 1792) #
    input_shape = (400, 320, 3)  # (1280, 1792, 3) #

    # paths
    base_folder = "../../datasets/hedychium_coronarium/"
    train_folder = base_folder + "train_splits/"
    augmentation_folder = train_folder + "aug/"
    validation_folder = base_folder + "val_splits/"
    test_folder = base_folder + "test/"
    image_folder = "images"
    label_folder = "masks"
    patience = 5

    tz = pytz.timezone("Brazil/East")
    start_time = datetime.now(tz=tz)
    path = datetime.now().strftime("%Y.%m.%d_%H%M%S")
    my_train_gene = None
    my_validation_gene = None
    flag_multi_class = True
    early_stopping_monitor = "accuracy"
    model_monitor = "accuracy"
    validation_steps = 200
    use_numpy = False

    def main(self, args):

        if args.t == -1:
            self.show_arguments()

        if args.t == 0:
            self.train(args)

        elif args.t == 1:
            self.test(args)

        elif args.t == 2:
            self.show_summary(args)

        elif args.t == 3:
            self.show_summary(args)

        elif args.t == 4:
            self.get_fbetaScore(args.b, args.p, args.r)

        elif args.t == 5:
            self.crop_images_in_tiles('../../datasets/hedychium_coronarium/', 
                                      'train',
                                      'val',
                                      'test',
                                      "images", 
                                      "masks", 
                                      400,
                                      320)

    def show_arguments(self):
        print("batch_size: ", self.batch_size)
        print("target_size: ", self.target_size)
        print("input_shape: ", self.input_shape)
        print("steps_per_epoch: ", self.steps_per_epoch)
        print("epochs: ", self.epochs)
        print("base_folder: ", self.base_folder)
        print("train_folder: ", self.train_folder)
        print("validation_folder: ", self.validation_folder)
        print("augmentation_folder: ", self.augmentation_folder)
        print("test_folder: ", self.test_folder)
        print("image_folder: ", self.image_folder)
        print("label_folder: ", self.label_folder)
        print("patience: ", self.patience)
        print("flag_multi_class: ", self.flag_multi_class)
        print("early_stopping_monitor: ", self.early_stopping_monitor)
        print("model_monitor: ", self.model_monitor)
        print("validation_steps: ", self.validation_steps)
        print("use_numpy: ", self.use_numpy)

    def set_arguments(
        self,
        batch_size=4,
        steps_per_epoch=50,
        epochs=15,
        target_size=(400, 320),
        input_shape=(400, 320, 3),
        base_folder="../hedychium_coronarium/",
        image_folder="images",
        label_folder="masks",
        patience=5,
        flag_multi_class=True,
        early_stopping_monitor="accuracy",
        model_monitor="accuracy",
        validation_steps=800,
        use_numpy = False
    ):
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.target_size = target_size
        self.input_shape = input_shape
        self.base_folder = base_folder
        self.train_folder = base_folder + "train_splits/"
        self.augmentation_folder = self.train_folder + "aug/"
        self.validation_folder = base_folder + "val_splits/"
        self.test_folder = base_folder + "test/"
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.patience = patience
        self.flag_multi_class = flag_multi_class
        self.early_stopping_monitor = early_stopping_monitor
        self.model_monitor = model_monitor
        self.class_weights = None
        self.validation_steps = validation_steps
        self.use_numpy = use_numpy

    def get_folder_name(self, basePath):
        now = datetime.now()
        self.path = now.strftime("%Y.%m.%d_%H%M%S")
        Path(basePath + self.path).mkdir(parents=True, exist_ok=True)
        return basePath + self.path

    def get_files_count(
        self,
        path,
        ext=".JPG",
        flag_multi_class=False,
        target_size=(256, 256),
        as_gray=False,
    ):
        parts = len(path.split("/"))
        imgs = glob.glob(path + "/*" + ext)

        for i, item in enumerate(imgs):
            imgs[i] = imgs[i].split("/")[parts]

        # imgs = glob.glob(path + '/*' + ext)

        # for item in enumerate(imgs):
        #     img = io.imread(item[1], as_gray = as_gray)
        #     img = img / 255.
        #     img = trans.resize(img, target_size)
        #     img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        #     img = np.reshape(img,(1,)+img.shape)
        #     yield img

        return len(imgs), imgs

    def arguments(
        self,
    ):
        # show options, get arguments and validate
        parser = argparse.ArgumentParser(description="Informe os parametros:")
        parser.add_argument(
            "--t",
            default=0,
            type=int,
            help="Informe o tipo  '--t -1' parametros, '--t 0' treino, '--t 1' teste, '--t 2' sumario', '--t 3' avaliacao, '--t 4' f-beta-score",
        )
        parser.add_argument(
            "--g",
            default=0,
            type=int,
            help="Gerar arquivos '--g 0' para nao gerar arquivos ou '--g 1' para gerar",
        )
        parser.add_argument(
            "--q",
            default=0,
            type=int,
            help="Quantidade de arquivos para teste '--q 0' para nao gerar arquivos ou '--q 1' para gerar",
        )
        parser.add_argument(
            "--n",
            default=None,
            type=str,
            help="Informe o nome do arquivo de pesos para executar o teste ou ler o sumario",
        )
        parser.add_argument(
            "--b",
            default=None,
            type=float,
            help="Informe o beta para calcular o f-beta score",
        )
        parser.add_argument(
            "--p",
            default=None,
            type=float,
            help="Informe o precision para calcular o f-beta score",
        )
        parser.add_argument(
            "--r",
            default=None,
            type=float,
            help="Informe o recall para calcular o f-beta score",
        )

        args = parser.parse_args()

        if args.t != -1 and args.t != 0 and args.t != 1 and args.t != 2 and args.t != 4:
            print(
                "Tipo invalido! Informe o tipo corretamente: --t -1' parametros, '--t 0' para treino, '--t 1' para teste', '--t 2' para exibir o sumario, '--t 4' f-beta-score"
            )
            sys.exit()

        if args.g != 0 and args.g != 1:
            print(
                "Parametro para a geracao de arquivos invalido! Informe corretamente: '--g 0' para nao gerar arquivos ou '--g 1' para gerar"
            )
            sys.exit()

        if (args.t == 2) and not args.n:
            print("Parametro invalido! Informe corretamente: '--n [file_name]'")
            sys.exit()

        if (args.t == 4) and (not args.b or not args.p or not args.r):
            print(
                "Parametro invalido! Informe corretamente: '--b [beta], --p [precision], --r [recall]'"
            )
            sys.exit()

        return args

    def generate_my_gen(self, args):
        data_gen_args = dict(
            rotation_range=0.2,
            zoom_range=0.05,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="wrap",
        )

        save_to_dir = None
        if args.g != 0:
            save_to_dir = self.get_folder_name(self.augmentation_folder)

        if (not self.use_numpy):
            self.my_train_gene = data_generator(
                self.batch_size,
                self.train_folder,
                self.image_folder,
                self.label_folder,
                data_gen_args,
                flag_multi_class=args.flag_multi_class,
                target_size=self.target_size,
                image_color_mode="rgb",
                save_to_dir=save_to_dir,
            )

            self.my_validation_gene = data_generator(
                self.batch_size,
                self.validation_folder,
                self.image_folder,
                self.label_folder,
                data_gen_args,
                flag_multi_class=args.flag_multi_class,
                target_size=self.target_size,
                image_color_mode="rgb",
                save_to_dir=save_to_dir)
            
            return (self.my_train_gene, self.my_validation_gene)

        else:
            self.my_train_gene_npy = gene_data_npy(self.train_folder, flag_multi_class=args.flag_multi_class)
            self.my_validation_gene_npy = gene_data_npy(self.validation_folder, flag_multi_class=args.flag_multi_class)

            return (self.my_train_gene_npy, self.my_validation_gene_npy)

    def train(self, args, generator_train, generator_val):

        # define TensorBoard directory and TensorBoard callback
        tb_cb = self.create_tensor_board_callback()

        try:
            self.show_execution_time(originalMsg="Starting now...", writeInFile=True)

            model = self.get_model()
            model.reset_metrics()

            if(not generator_train or not generator_val):
                (generator_train, generator_val) = self.generate_my_gen(args)

            earlystopper = EarlyStopping(
                patience=self.patience,
                verbose=1,
                monitor=self.early_stopping_monitor,
                mode="auto",
            )

            model_checkpoint = ModelCheckpoint(
                f"train_weights/{self.path}_unet.hdf5",
                monitor=self.model_monitor,
                verbose=0,
                save_best_only=True,
            )

            # classe desbalanceada
            # self.class_weights = { 0: 0.80, 1: 0.20}

            model.summary()

            model.fit(
                generator_train,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                validation_data=generator_val,
                validation_steps=self.validation_steps,
                callbacks=[earlystopper, model_checkpoint, tb_cb],
            )

            self.show_execution_time(writeInFile=True)
            model.save_weights(f"train_weights/final_{self.path}_unet.hdf5")

        except Exception as e:
            self.show_execution_time(success=False, writeInFile=True)
            error_Msg = (
                "\ntype error: " + str(e) + " \ntraceback: " + traceback.format_exc()
            )
            self.show_execution_time(
                success=False, originalMsg=error_Msg, writeInFile=True
            )
            raise e

    def get_model(self):
        return unet(pretrained_weights=None, input_size=self.input_shape)

    def evaluate(self, model: Model, X, Y):
        model.evaluate(x=X, y=Y)

    def test(self, args):

        if not args.n:
            args.n = "train_weights/20200420_0817_unet-100-100-loss0_431_acc0_9837.hdf5"
        else:
            args.n = "train_weights/" + args.n

        qtd, imgs = self.get_files_count(
            self.test_folder + self.image_folder, target_size=self.target_size
        )

        if qtd > 0:
            try:
                self.show_execution_time(
                    originalMsg="Starting now...", writeInFile=True
                )

                tb_cb = self.create_tensor_board_callback()
                testGene = test_generator(
                    self.test_folder + self.image_folder + "/",
                    flag_multi_class=args.flag_multi_class,
                    target_size=self.input_shape,
                    as_gray=False,
                )

                model = unet(pretrained_weights=args.n, input_size=self.input_shape)

                results = model.predict(
                    testGene, steps=qtd, callbacks=[tb_cb], verbose=1
                )

                save_result(
                    save_path=self.test_folder + "/results",
                    npyfile=results,
                    imgs=imgs,
                    flag_multi_class=args.flag_multi_class,
                )

                # res = model.evaluate(x=results, verbose=0, callbacks=[tb_cb])
                # res = model.predict(x=my_gene, batch_size=batch_size, callbacks=[CustomCallback()])

                self.show_execution_time(writeInFile=True)

            except Exception as e:
                self.show_execution_time(success=False, writeInFile=True)
                error_Msg = (
                    "\ntype error: "
                    + str(e)
                    + " \ntraceback: "
                    + traceback.format_exc()
                )
                print(error_Msg)
                self.show_execution_time(
                    success=False, originalMsg=error_Msg, writeInFile=True
                )
                pass

        else:
            print("nenhum arquivo encontrado")

    def show_summary(self, args):
        model = unet(pretrained_weights=args.n, input_size=self.input_shape)
        model.build(self.input_shape)
        model.summary()

    def show_execution_time(self, success=True, originalMsg="", writeInFile=False):
        end_time = datetime.now(tz=self.tz)
        elapsed = end_time - self.start_time

        msg = (
            originalMsg + f"\n=================================================="
            f"\nExecution checkpoint! Success: {success} "
            f'\nProcess Started: {self.start_time.strftime("%Y/%m/%d %H:%M")}'
            f'\nProcess Now: {end_time.strftime("%Y/%m/%d %H:%M")}'
            f"\nProcess Time elapsed: {elapsed}"
            f"\n=================================================="
            f"\n\n\n"
        )
        if writeInFile:
            basePath = f'.logs/{self.start_time.strftime("%Y%m%d")}/'
            path = f'{self.start_time.strftime("%Y%m%d_%H%M")}/'
            self.write_text_file(basePath, path, "execution_log.txt", msg)

    def write_text_file(self, basePath, path, file_name, text):
        self.create_directory(basePath, path)

        text_file = open(basePath + path + file_name, "a+")
        n = text_file.write(text)
        text_file.close()

    def create_directory(self, basePath, path):
        if basePath and not os.path.exists(basePath):
            os.makedirs(basePath)

        if basePath + path and not os.path.exists(basePath + "/" + path):
            os.makedirs(basePath + "/" + path)

    def create_tensor_board_callback(self):
        basePath = f'.logs/{self.start_time.strftime("%Y%m%d")}'
        path = self.start_time.strftime("%Y%m%d_%H%M")
        tb_dir = f"{basePath}/{path}/"
        tb_cb = TensorBoard(log_dir=tb_dir, write_graph=True, update_freq=1)

        self.create_directory(basePath, path)

        return tb_cb

    def get_recall(self, tp, fn):
        recall = tp / (tp + fn)
        print(f"The recall value is: {recall}")
        return recall

    def get_precision(self, tp, tn):
        precision = tp / (tp + tp)
        print(f"The precision value is: {precision}")
        return precision

    def get_f1_measure(
        self,
    ):
        return 0

    def crop_images_in_tiles(self, data_path, train_folder, val_folder, test_folder, image_folder, mask_folder, w, h):
        if(train_folder):
            crop_image(data_path, train_folder, image_folder, mask_folder , w, h, True)
        if(val_folder):
            crop_image(data_path, val_folder, image_folder, mask_folder , w, h)
        if(test_folder):
            crop_image(data_path, test_folder, image_folder, mask_folder , w, h)

        self.move_ignored_items(data_path)

    def move_ignored_items(self, data_path):
        move_all_ignored_folders_to_test(data_path)

    def get_fscore(self, recall, precision):
        f_score = (2 * recall * precision) / (recall + precision)
        print(f"The f_score value is: {f_score}")
        return f_score

    def get_fbetaScore(self, beta, precision, recall):

        # F0.5-Measure  (beta=0.5): More weight on precision, less weight on recall
        # F1-Measure    (beta=1.0): Balance the weight on precision and recall
        # F2-Measure    (beta=2.0): Less weight on precision, more weight on recall
        f_beta_score = ((1 + beta ** 2) * precision * recall) / (
            beta ** 2 * precision + recall
        )
        print(f"The f_beta_score value is: {f_beta_score}")
        return f_beta_score

    def plot_roc_curve(self, fpr, tpr):
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()

    def get_auc(self, model, testX, testY):
        probs = model.predict_proba(testX)
        probs = probs[:, 1]

        auc = roc_auc_score(testY, probs)
        print("AUC: %.2f" % auc)

        fpr, tpr, thresholds = roc_curve(testY, probs)
        self.plot_roc_curve(fpr, tpr)

        return auc

    # Jaccard Index
    def iou_coef(self, y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
