from unet_helper import UnetHelper

if __name__ == "__main__":     

    helper = UnetHelper()
    args = helper.arguments()

    helper.main()