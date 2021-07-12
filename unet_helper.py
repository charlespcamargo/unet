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
import tensorflow as tf
from tensorflow.keras.callbacks import *

class UnetHelper:
    # training vars
    model = None
    batch_size = 1
    steps_per_epoch = 20
    epochs = 3

    # image sizes
    target_size = (256, 256)   
    input_shape = (256, 256, 3)  

    # paths
    base_folder = "/Users/charles/Downloads/hedychium_coronarium/versions/v2/hedychium_coronarium/"
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
    early_stopping_monitor = "val_auc"
    early_stopping_monitor_mode = "max"
    class_weight = None
    model_monitor = "val_auc"
    model_monitor_mode = "max"    
    validation_steps = 10
    use_numpy = False
    learning_rate = 1e-4
    momentum = 0.90
    use_sgd = False
    use_euclidean = True
    check_train_class_weights = False
    use_augmentation = False
    use_splits = False

    def main(self, args):

        #self.playground()
        #return False

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
                                      256,
                                      256,
                                      threshold = 20,
                                      force_delete = False)

        elif args.t == 6:
            PreProcessingData.crop_all_images_in_tiles('/Users/charles/Downloads/hedychium_coronarium/all', 
                                      "images", 
                                      "masks", 
                                      256,
                                      256,
                                      threshold = 25,
                                      force_delete = False,
                                      validate_class_to_discard = True,
                                      move_ignored_to_test = False)

        elif args.t == 7:            
            PreProcessingData.get_train_class_weights('../../datasets/all', use_splits=self.use_splits)

        elif args.t == 8:
            self.compare_result()

    def playground(self):
        import PIL
        from PIL import Image
        qtd, imgs = self.get_files_count(self.train_folder + self.image_folder)
        
        for i, file_name in enumerate(imgs):            
            img = io.imread(os.path.join(self.train_folder, "images", file_name), as_gray=False)
            mask = io.imread(os.path.join(self.train_folder, "masks", file_name), as_gray=False)
            
            #y_pred_fake = self.generate_fake_predict(mask)
            y_pred_fake = io.imread(os.path.join(self.train_folder, "masks", file_name), as_gray=False)

            y_pred_fake2 = CustomMetricsAndLosses.weighted_bce_dice_loss(mask, y_pred_fake)
            y_pred_fake2 = y_pred_fake2.numpy()           
            
            w = 256
            h = 256
            composed_img = Image.new('RGB', (w*2, h*2), color="gray")
            composed_img.paste(Image.fromarray(img, 'RGB'), (0, 0))
            composed_img.paste(Image.fromarray(mask, 'RGB'), (w, 0))
            composed_img.paste(Image.fromarray(img, 'RGB'), (0, h))
            composed_img.paste(Image.fromarray(y_pred_fake, 'RGB'), (w, h))
            composed_img.show()            

            if(i>10):
                break

            
    def generate_fake_predict(self, mask):
        y_pred_fake = np.random.rand(256, 256, 3) * mask
        #y_pred_fake = y_pred_fake.astype('float32')
        #y_pred_fake = y_pred_fake / 255
        y_pred_fake[y_pred_fake >= 127.5] = 255
        y_pred_fake[y_pred_fake < 127.5] = 0
        y_pred_fake = np.abs(np.mean(y_pred_fake, axis=2) > 0.5) * 255
        # y_pred_fake_true = int((y_pred_fake == 255).sum())
        # y_pred_fake_false = int((y_pred_fake == 0).sum())

        # total = y_pred_fake_true + y_pred_fake_false
        # print(f'fator: {fator} - false: {y_pred_fake_false}({round(y_pred_fake_false/total*100, 2)}%) - true: {y_pred_fake_true}({round(y_pred_fake_true/total*100, 2)}%) - total: {total}')

        # mask_true = int((mask == 255).sum())
        # mask_false = int((mask == 0).sum())
        # diff = mask.sum() + mask_true + mask_false
        # total = mask_true + mask_false
        # print(f'false: {mask_false}({round(mask_false/total*100, 2)}%) - true: {mask_true}({round(mask_true/total*100, 2)}%) - total: {total}')
        
        return y_pred_fake


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
        print("use_euclidean:", self.use_euclidean)
        print("check_train_class_weights", self.check_train_class_weights)
        print("use_augmentation", self.use_augmentation)
        print("use_splits", self.use_splits)


    def set_arguments(
        self,
        batch_size=2,
        steps_per_epoch=50,
        epochs=15,
        target_size=(256, 256),
        input_shape=(256, 256, 3),
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
        use_euclidean = False,
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
        self.use_euclidean = use_euclidean
        self.check_train_class_weights = check_train_class_weights
        self.use_augmentation = use_augmentation
        self.use_splits = use_splits

    def get_folder_name(self, base_path):
        now = datetime.now()
        self.path = now.strftime("%Y%m%d_%H%M%S")
        Path(base_path).mkdir(parents=True, exist_ok=True)
        return base_path

    def get_files_count(
        self,
        path,
        ext=".JPG"
    ):
        parts = len(path.split("/"))
        imgs = glob.glob(path + "/*" + ext)

        for i, item in enumerate(imgs):
            imgs[i] = imgs[i].split("/")[parts]

        return len(imgs), imgs


    def get_some_weight(
        self,
        path='train_weights/',
        ext='.hdf5'
    ):
        files = glob.glob(path + "/*" + ext)
        total = len(files)
        last_file_aux = 0
        last_file = None
        
        for i, item in enumerate(files):
            print(f'{i}/{total-1} - {item}')
            arr = item.replace('/','_').split('_')
            file_name_date = int(arr[len(arr)-3] + arr[len(arr)-2])
            
            if(not last_file or file_name_date > int(last_file_aux)):
                last_file_aux = file_name_date
                last_file = item

        return last_file
    
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
                                    zoom_range = [0.01, 0.25],  # alterar
                                    brightness_range=[0.5, 1.5], # alterar
                                    shear_range= 10.0,
                                    horizontal_flip=True, # remover
                                    vertical_flip=True, # remover            
                                    fill_mode="wrap"# remover        
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

        # just for debugging
        #tf.config.run_functions_eagerly(True)

        try:
            self.show_execution_time(original_msg="Starting now...", write_in_file=True)

            model = self.get_model(pretrained_weights=args.n)

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
                save_weights_only=False
            )

            # classe desbalanceada
            if(self.check_train_class_weights):
                (black, white) = PreProcessingData.get_train_class_weights(self.base_folder)
                self.class_weights = { 0: white }
                print(f"self.class_weights: {self.class_weights}")

            history = model.fit(
                x=generator_train,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                validation_data=generator_val,
                validation_steps=self.validation_steps,
                callbacks=[earlystopper, model_checkpoint, tb_cb],
                verbose=1,
                class_weight=self.class_weight
            )

            self.show_execution_time(write_in_file=True)
            
            self.evaluate(args, model, history)

            return history

        except Exception as e:
            self.show_execution_time(success=False, write_in_file=True)
            error_msg = (
                "\ntype error: " + str(e) + " \ntraceback: " + traceback.format_exc()
            )
            self.show_execution_time(
                success=False, original_msg=error_msg, write_in_file=True
            )
            raise e 

    def get_model(self, pretrained_weights = None, cnn_type = 0):
        unet = Unet()

        return unet.create_model(pretrained_weights=pretrained_weights, input_size=self.input_shape, num_class=2, 
                                  learning_rate = self.learning_rate, momentum = self.momentum, use_sgd = self.use_sgd, use_euclidean = self.use_euclidean)

    def evaluate(self, args, model: Model, history, steps_to_test = 100):

        print('Evaluating model in test DB...')
        total, imgs = self.get_files_count(self.test_folder + self.image_folder)
        
        steps_to_test = total if (steps_to_test <= 0) else steps_to_test
        total = total if (total < steps_to_test) else steps_to_test
        page_size = 100                 
        page_size = total if (page_size > total) else page_size
        pages = math.ceil(total / page_size) 
        current_page = 0

        if total > 0:
            for current_page in range(0, pages):
                page_size = total if (page_size > total) else page_size
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

            acc = model.evaluate(test_gene, verbose=1)
            print('Evaluate data - acc: %.3f - and plotting...', acc)
            
            # plot training history
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_binary_accuracy'], label='test')
            plt.legend()
            plt.show()
        
        else:
            print("nenhum arquivo encontrado") 

    def test(self, args, steps_to_test = 0, cnn_type = 0):

        if not args.n:
            print('No weight was informed! trying to fetch some weight')
            file_name = self.get_some_weight()

            if not file_name:
                print('No weight was found!')
                return False
            else:
                print(f'Using the fetched file weight found: {file_name}')
                args.n = file_name
    

        total, imgs = self.get_files_count(self.test_folder + self.image_folder)

        if total > 0:
            try:
                self.show_execution_time(
                    original_msg="Starting now...", write_in_file=True
                )

                steps_to_test = total if (steps_to_test <= 0) else steps_to_test
                total = total if (total < steps_to_test) else steps_to_test
                page_size = 20                 
                page_size = total if (page_size > total) else page_size
                pages = math.ceil(total / page_size) 
                current_page = 0
                results = {}

                tb_cb = self.create_tensor_board_callback()
                model = self.get_model(pretrained_weights=args.n, cnn_type = cnn_type)
                
                for current_page in range(0, pages):
                    page_size = total if (page_size > total) else page_size
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
                    
                
                self.show_execution_time(write_in_file=True)

            except Exception as e:
                self.show_execution_time(success=False, write_in_file=True)
                error_msg = (
                    "\ntype error: "
                    + str(e)
                    + " \ntraceback: "
                    + traceback.format_exc()
                )
                print(error_msg)
                self.show_execution_time(
                    success=False, original_msg=error_msg, write_in_file=True
                )

        else:
            print("nenhum arquivo encontrado") 

    def compare_result(self,):
        qtd, imgs = self.get_files_count(self.test_folder + self.image_folder)
        Data.compare_result(self.test_folder, imgs)

    def show_summary(self, args):
        unet = Unet()
        model = unet.create_model(pretrained_weights=args.n, input_size=self.input_shape)
        model.build(self.input_shape)
        model.summary()

    def show_execution_time(self, success=True, original_msg="", write_in_file=False):
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
        if write_in_file:
            base_path = f'.logs/{self.start_time.strftime("%Y%m%d")}/'
            path = f'{self.start_time.strftime("%Y%m%d_%H%M")}/'
            self.write_text_file(base_path, path, "execution_log.txt", msg)

    def write_text_file(self, base_path, path, file_name, text):
        self.create_directory(base_path, path)

        text_file = open(base_path + path + file_name, "a+")
        text_file.write(text)
        text_file.close()

    def create_directory(self, base_path, path):
        if base_path and not os.path.exists(base_path):
            os.makedirs(base_path)

        if base_path + path and not os.path.exists(base_path + "/" + path):
            os.makedirs(base_path + "/" + path)

    def create_tensor_board_callback(self):
        base_path = f'.logs/{self.start_time.strftime("%Y%m%d")}'
        path = self.start_time.strftime("%Y%m%d_%H%M")
        tb_dir = f"{base_path}/{path}/"
        tb_cb = TensorBoard(log_dir=tb_dir, write_graph=True, update_freq=1)

        self.create_directory(base_path, path)

        return tb_cb

