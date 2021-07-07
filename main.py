from unet_helper import UnetHelper

if __name__ == "__main__":     
    
    import easydict
    myhelper = UnetHelper()

    args = easydict.EasyDict({
        't': 0, # "Informe o tipo  
         # '--t -1' parametros, '--t 0' treino, '--t 1' teste, '--t 2' sumario', '--t 3' avaliacao, '--t 4' f-beta-score, '--t 5' crop_images", 
         # '--t 5' crop_images_in_tiles", '--t 6' crop_all_images_in_tiles", '--t 7' get_train_class_weights", '--t 8' compare_result"
        'g': 1, # Gerar arquivos '--g 0' para nao gerar arquivos ou '--g 1' para gerar
        'q': 0, # Quantidade de arquivos para teste '--q 0' para nao gerar arquivos ou '--q 1' para gerar
        'n': '20210701_191701_unet.hdf5', # Informe o nome do arquivo de pesos para executar o teste ou ler o sumario
        'b': None, # Informe o beta para calcular o f-beta score
        'p': None, # Informe o precision para calcular o f-beta score
        'r': None, # Informe o recall para calcular o f-beta score,
        'flag_multi_class': False,
        'batch_size': 1, 
        'steps_per_epoch': 20, 
        'epochs': 5, 
        'target_size': (256, 256), 
        'input_shape': (256, 256, 3),
        'base_folder': '/Users/charles/Downloads/hedychium_coronarium/', 
        'image_folder': 'images', 
        'label_folder': 'masks', 
        'patience': 10,
        'early_stopping_monitor': 'val_acc', 
        'early_stopping_monitor_mode': 'max', 
        'class_weights': None,
        'model_monitor': 'val_auc', 
        'model_monitor_mode': 'max', 
        'validation_steps': 200, 
        'use_numpy': False,
        'learning_rate': 1e-4,
        'momentum': 0.90
    })

    myhelper.main(args) 