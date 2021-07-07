import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *  
from tensorflow.python.keras.backend import backend, sigmoid, softmax
from custom_metrics import *


class Unet():

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, pretrained_weights = None, input_size = (256,256, 3), num_class = 2, learning_rate = 1e-4, momentum = 0.90, use_sgd = False, use_euclidean = False):
        inputs = Input( shape=(input_size) ) 

        concat_axis = 3  
        # pesquisar camada de processamento
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(inputs)    
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv1)    
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis=concat_axis)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis=concat_axis)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis=concat_axis)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis=concat_axis)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dtype=tf.float32)(conv9)
        print("conv9 shape:", conv9.shape) 

        ##Output Layer
        output_layer = Conv2D(3, 1, activation = 'sigmoid', dtype=tf.float32)(conv9)

        print(output_layer)

        ##Defining Model
        model = Model(inputs=inputs, outputs=output_layer)
        opt = Adam(learning_rate = learning_rate)
        loss = BinaryCrossentropy(name='binary_crossentropy')

        if(use_sgd == True):
            opt = SGD(learning_rate = learning_rate, momentum = momentum)            

        if(use_euclidean == True):
            loss = CustomMetrics.euclidean_distance_loss

        ##Compiling Model 
        model.compile(optimizer = opt, 
                     loss = loss,
                     metrics = [
                                Precision(name="precision"),
                                Recall(name="recall"),
                                AUC(name="auc"),
                                MeanIoU(num_classes=2, name="mean_iou"),
                                Accuracy(name="accuracy"),
                                BinaryAccuracy(name="binary_accuracy"),                               
                                CustomMetrics.jacard_coef,
                                CustomMetrics.dice_coefficient
                            ]
                )

                             

        model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        return model