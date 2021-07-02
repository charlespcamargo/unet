from model import *
from data import *
from custom_metrics import *
from pre_processing_data import *

from pathlib import Path
import sys
import argparse
import os
import os.path
import glob
import traceback
import math

from datetime import datetime
import pytz

# import seaborn as sns

import tensorflow as tf

# import tensorflow.keras
from tensorflow.keras.callbacks import *
from skimage.util.dtype import img_as_float32, img_as_uint


class UnetHelper:
    # training vars
    model = None
    batch_size = 1
    steps_per_epoch = 5
    epochs = 1

    # image sizes
    target_size = (416, 416)  # (1280, 1792) #
    input_shape = (416, 416, 3)  # (1280, 1792, 3) #

    # paths
    base_folder = "/Users/charles/Downloads/hedychium_coronarium/"
    train_folder = base_folder + "train/"
    augmentation_folder = train_folder + "aug/"
    validation_folder = base_folder + "val/"
    test_folder = base_folder + "test/"
    image_folder = "images"
    label_folder = "masks"
    patience = 5

    tz = pytz.timezone("Brazil/East")
    start_time = datetime.now(tz=tz)
    path = datetime.now().strftime("%Y.%m.%d_%H%M%S")
    my_train_gene = None
    my_validation_gene = None
    flag_multi_class = False
    early_stopping_monitor = "val_mean_iou"
    early_stopping_monitor_mode = "auto"
    class_weight = None
    model_monitor = "val_binary_accuracy"
    model_monitor_mode = "auto"    
    validation_steps = 10
    use_numpy = False
    learning_rate = 1e-4
    momentum = 0.90
    use_sgd = False
    check_train_class_weights = False
    use_augmentation = False
    use_splits = False

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
            PreProcessingData.crop_images_in_tiles('../../datasets/hedychium_coronarium/', 
                                      'train',
                                      'val',
                                      'test',
                                      "images", 
                                      "masks", 
                                      416,
                                      416,
                                      threshold = 20,
                                      force_delete = False)

        elif args.t == 6:
            PreProcessingData.crop_all_images_in_tiles('/Users/charles/Downloads/hedychium_coronarium/all', 
                                      "images", 
                                      "masks", 
                                      416,
                                      416,
                                      threshold = 40,
                                      force_delete = False,
                                      validate_class_to_discard = True,
                                      move_ignored_to_test = False)

        elif args.t == 7:            
            PreProcessingData.get_train_class_weights('../../datasets/all', use_splits=self.use_splits)

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
        print("early_stopping_monitor_mode: ", self.early_stopping_monitor_mode)
        print("class_weight: ", self.class_weight)
        print("model_monitor: ", self.model_monitor)
        print("model_monitor_mode: ", self.model_monitor_mode)
        print("validation_steps: ", self.validation_steps)
        print("use_numpy: ", self.use_numpy)
        print("learning_rate: ", self.learning_rate)
        print("momentum:", self.momentum)
        print("use_sgd:", self.use_sgd)
        print("check_train_class_weights", self.check_train_class_weights)
        print("use_augmentation", self.use_augmentation)
        print("use_splits", self.use_splits)


    def set_arguments(
        self,
        batch_size=2,
        steps_per_epoch=50,
        epochs=15,
        target_size=(416, 416),
        input_shape=(416, 416, 3),
        base_folder="../hedychium_coronarium/",
        image_folder="images",
        label_folder="masks",
        patience=5,
        flag_multi_class=False,
        early_stopping_monitor="val_mean_iou",
        early_stopping_monitor_mode ="auto",
        model_monitor = "val_binary_accuracy",
        model_monitor_mode = "auto",
        class_weights = None,
        validation_steps=200,
        use_numpy = False,
        learning_rate = 1e-4,
        momentum = 0.90,
        use_sgd = False,
        check_train_class_weights = False,
        use_augmentation = False,
        use_splits = False,
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
        self.test_folder = base_folder + "test_splits/"
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.patience = patience
        self.flag_multi_class = flag_multi_class
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_monitor_mode = early_stopping_monitor_mode
        self.model_monitor = model_monitor
        self.model_monitor_mode = model_monitor_mode
        self.class_weights = class_weights
        self.validation_steps = validation_steps
        self.use_numpy = use_numpy
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_sgd = use_sgd
        self.check_train_class_weights = check_train_class_weights
        self.use_augmentation = use_augmentation
        self.use_splits = use_splits

    def get_folder_name(self, basePath):
        now = datetime.now()
        self.path = now.strftime("%Y%m%d_%H%M%S")
        Path(basePath).mkdir(parents=True, exist_ok=True)
        return basePath

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
        
        data_gen_args = dict()
        
        if(self.use_augmentation):
            data_gen_args = dict(
                                    zoom_range=0.1,  # alterar
                                    brightness_range=[0.25,0.25], # alterar
                                    #width_shift_range=0.0, # remover
                                    #height_shift_range=0.0, # remover
                                    #shear_range=1.0, # remover
                                    #rotation_range=90,  # remover
                                    #horizontal_flip=True, # remover
                                    #vertical_flip=True, # remover            
                                    #fill_mode="wrap"# remover        
                                )
        else:
            print('without augmentation')

        save_to_dir = None
        if args.g != 0:
            save_to_dir = self.get_folder_name(self.augmentation_folder)

        if (not self.use_numpy):
            self.my_train_gene = Data.data_generator(
                self.batch_size,
                self.train_folder,
                self.image_folder,
                self.label_folder,
                data_gen_args,
                flag_multi_class=args.flag_multi_class,
                target_size=self.target_size,
                image_color_mode="rgb",
                save_to_dir=save_to_dir
            )

            self.my_validation_gene = Data.data_generator(
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
            self.my_train_gene_npy = Data.gene_data_npy(self.train_folder, flag_multi_class=args.flag_multi_class)
            self.my_validation_gene_npy = Data.gene_data_npy(self.validation_folder, flag_multi_class=args.flag_multi_class)

            return (self.my_train_gene_npy, self.my_validation_gene_npy)

    def train(self, args, generator_train = None, generator_val = None):

        # define TensorBoard directory and TensorBoard callback
        tb_cb = self.create_tensor_board_callback()

        # to improve speed for some gpus
        # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        # tf.keras.mixed_precision.experimental.set_policy(policy) 

        try:
            self.show_execution_time(original_msg="Starting now...", writeInFile=True)

            model = self.get_model()
            #model.reset_metrics()

            if(not generator_train or not generator_val):
                (generator_train, generator_val) = self.generate_my_gen(args)

            earlystopper = EarlyStopping(
                patience=self.patience,
                verbose=1,
                monitor=self.early_stopping_monitor,
                mode=self.early_stopping_monitor_mode,
            )

            model_checkpoint = ModelCheckpoint(
                f"train_weights/{self.path}_unet.hdf5",
                monitor=self.model_monitor,
                mode=self.model_monitor_mode,
                verbose=1,
                save_best_only=True,
                save_weights_only=True
            )

            # classe desbalanceada
            if(self.check_train_class_weights):
                (black, white) = PreProcessingData.get_train_class_weights(self.base_folder)
                self.class_weights = { 0: white }
                print(f"self.class_weights: {self.class_weights}")

            #model.summary()

            history = model.fit(
                generator_train,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                validation_data=generator_val,
                validation_steps=self.validation_steps,
                callbacks=[earlystopper, model_checkpoint, tb_cb],
                verbose=1,
                class_weight=self.class_weight
            )

            #print('Evaluating train...')
            #self.evaluate(model, generator_train, history)
            
            #print('Evaluating val...')
            #self.evaluate(model, generator_val, history)

            self.show_execution_time(writeInFile=True)
            #model.save_weights(f"train_weights/final_{self.path}_unet.hdf5")

        except Exception as e:
            self.show_execution_time(success=False, writeInFile=True)
            error_msg = (
                "\ntype error: " + str(e) + " \ntraceback: " + traceback.format_exc()
            )
            self.show_execution_time(
                success=False, original_msg=error_msg, writeInFile=True
            )
            raise e 

    def get_model(self, pretrained_weights = None, cnn_type = 0):
        unet = Unet()

        if(cnn_type == 0):
            return unet.create_model(pretrained_weights=pretrained_weights, input_size=self.input_shape, num_class=2, learning_rate = self.learning_rate, momentum = self.momentum, use_sgd = self.use_sgd)
        elif(cnn_type == 1):
            return unet.create_model_keras(img_size=self.input_shape, num_classes=2)        
        else:
            return unet.create_model_zizhaozhang(input_size = self.input_shape, num_class = 2)

        #return unet.get_unet(self.input_shape, n_filters = 16, dropout = 0.1, batchnorm = True)

    def evaluate(self, model: Model, data_generator, history):
        _, acc = model.evaluate(data_generator, verbose=1)
        print('Evaluate data - acc: %.3f - and plotting...', acc)
        
        # plot training history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_binary_accuracy'], label='test')
        plt.legend()
        plt.show()


    def test2(self, args):
        args.n = "train_weights/20210701_191701_unet.hdf5"
        model = self.get_model(pretrained_weights=args.n, cnn_type = 0)
        imgs = glob.glob(self.test_folder + self.image_folder + "/*.JPG")
        
        output_layer_name = model.layers[len(model.layers)-1].name
        layer_output = model.get_layer(output_layer_name).output
        intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=layer_output)

        for item in imgs:
            img_original = io.imread(item, as_gray=False)
            x = item.split('/')
            l = len(x)
            x[l-2] = 'masks'
            mask_image =  os.path.join("/".join(x))
            img_mask = io.imread(mask_image, as_gray=False)
            img = img_original
            img = img.astype('float32')
            img = img / 255
            img = trans.resize(img, self.target_size)
            img = np.reshape(img, img.shape + (1,)) if (not False) else img
            img = np.reshape(img, (1,) + img.shape)                 

            # normalizando a saida
            intermediate_prediction = intermediate_model.predict(img.reshape(1, 416, 416,3))        
            prediction_binary = (np.mean(intermediate_prediction[0], axis=2) > 0.5) * 255
            
            w = 1500
            h = 1500 
            my_dpi = 96
            #fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
            
            #fig = plt.figure()
            #fig.add_subplot(2, 2, 1)

            fig, ax = plt.subplots(nrows=2, ncols=2)            
            ax[0][0].imshow(img_original)
            ax[0][0].set_title(f'original: {x[l-1]}')
            
            ax[0][1].imshow(img_mask)
            ax[0][1].set_title('mask')

            ax[1][0].imshow(prediction_binary, cmap='gray')
            ax[1][0].set_title('predict')
            
            img_mask = img_mask[:,:,0]
            diff = np.abs(img_mask - prediction_binary)
            ax[1][1].imshow(diff, cmap='summer')
            ax[1][1].set_title('diff') 
            

            plt.show() 

            # plt.imshow(prediction_binary, cmap='gray')
            # plt.title('predict') 

            # fig.add_subplot(2, 2, 2)
            # img_mask = img_mask[:,:,0]
            # diff = np.abs(img_mask - prediction_binary)
            # plt.imshow(diff, cmap='Purples')
            # plt.title('diff') 
            
            # fig.add_subplot(1, 2, 1)
            # plt.imshow(img_original)
            # plt.title(f'original: {x[l-1]}')

            # fig.add_subplot(1, 2, 2)
            # plt.imshow(img_mask)
            # plt.title('mask')

            # fig.tight_layout()    
            # plt.show() 


    def test(self, args, steps_to_test = 0, cnn_type = 0):

        if not args.n:
            args.n = "train_weights/20210701_191701_unet.hdf5"
        else:
            args.n = "train_weights/" + args.n

        qtd, imgs = self.get_files_count(
            self.test_folder + self.image_folder, target_size=self.target_size
        )

        if qtd > 0:
            try:
                self.show_execution_time(
                    original_msg="Starting now...", writeInFile=True
                )

                tb_cb = self.create_tensor_board_callback()
                model = self.get_model(pretrained_weights=args.n, cnn_type = cnn_type)

                steps_to_test = qtd if (steps_to_test <= 0) else steps_to_test
                total = qtd if (qtd < steps_to_test) else steps_to_test
                page_size = 20                 
                page_size = qtd if (page_size > qtd) else page_size
                pages = math.ceil(total / page_size) 
                current_page = 0
                results = {}

                for current_page in range(0, pages):
                    
                    page_size = qtd if (page_size > qtd) else page_size
                    if(page_size > 0):
                        offset = current_page * page_size
                        current_page_imgs = np.array(imgs)[offset : offset + page_size]
                    else:
                        current_page_imgs = imgs


                    test_gene = Data.test_generator(
                        current_page_imgs,
                        self.test_folder + self.image_folder + "/",
                        flag_multi_class = args.flag_multi_class,
                        target_size = self.input_shape,
                        as_gray = False
                    )

                    results = model.predict(test_gene, steps=len(current_page_imgs), batch_size=self.batch_size, max_queue_size=10, callbacks=[tb_cb], verbose=1,use_multiprocessing=False)
                    
                    Data.save_result(
                        save_path=self.test_folder + "results",
                        npyfile=results,
                        imgs=current_page_imgs,
                        flag_multi_class=False,
                    )   

                #res = model.evaluate(x=results, verbose=1, callbacks=[tb_cb])
                #self.evaluate(model, testGene, history)

                self.show_execution_time(writeInFile=True)

            except Exception as e:
                self.show_execution_time(success=False, writeInFile=True)
                error_msg = (
                    "\ntype error: "
                    + str(e)
                    + " \ntraceback: "
                    + traceback.format_exc()
                )
                print(error_msg)
                self.show_execution_time(
                    success=False, original_msg=error_msg, writeInFile=True
                )
                pass

        else:
            print("nenhum arquivo encontrado")

    def show_summary(self, args):
        unet = Unet()
        model = unet.create_model(pretrained_weights=args.n, input_size=self.input_shape)
        model.build(self.input_shape)
        model.summary()

    def show_execution_time(self, success=True, original_msg="", writeInFile=False):
        end_time = datetime.now(tz=self.tz)
        elapsed = end_time - self.start_time

        msg = (
            original_msg + f"\n=================================================="
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

