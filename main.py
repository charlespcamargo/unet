from unet_helper import UnetHelper

if __name__ == "__main__":     
    
    import easydict
    myhelper = UnetHelper()
    #args = helper.arguments()

    args = easydict.EasyDict({
        't': 1, # "Informe o tipo  '--t -1' parametros, '--t 0' treino, '--t 1' teste, '--t 2' sumario', '--t 3' avaliacao, '--t 4' f-beta-score")
        'g': 0, # Gerar arquivos '--g 0' para nao gerar arquivos ou '--g 1' para gerar
        'q': 0, # Quantidade de arquivos para teste '--q 0' para nao gerar arquivos ou '--q 1' para gerar
        'n': None, # Informe o nome do arquivo de pesos para executar o teste ou ler o sumario
        'b': None, # Informe o beta para calcular o f-beta score
        'p': None, # Informe o precision para calcular o f-beta score
        'r': None # Informe o recall para calcular o f-beta score
    })

    myhelper.main(args)