import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import *  
from tensorflow.keras import backend as K
from tensorflow.python.keras import regularizers
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

    def create_model(self, pretrained_weights = None, input_size = (256,256, 3), num_class = 2):
        inputs = Input( shape=input_size )
        concat_axis = 3 

        #https://stackoverflow.com/questions/53975502/where-to-add-kernal-regularizers-in-an-u-net
        #c1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(w_decay))) (s)
        #c1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(w_decay))) (c1)
        #, kernel_regularizer=regularizers.l2(w_decay))

        w_decay = 0.00009 # weight decay coefficient

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(inputs)    
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv1)    
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis=concat_axis)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis=concat_axis)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis=concat_axis)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis=concat_axis)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(w_decay))(conv9)
        
        ##Output Layer
        conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

        ##Defining Model
        model = Model(inputs, conv10)

        ##Compiling Model
        model.compile(optimizer = Adam(learning_rate = 1e-4), 
                      loss = 'binary_crossentropy',
                            # 'binary_crossentropy',
                      metrics = [BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                                 Precision(name="precision"),
                                 Recall(name="recall"),
                                 MeanIoU(num_classes=2, name="mean_iou"),
                                 AUC(name="auc"),
                                 CustomMetrics.jacard_coef,
                                 CustomMetrics.dice_coefficient]
                    )

                             

        model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def create_model_zizhaozhang(self, pretrained_weights = None, input_size = (256,256, 3), num_class = 2):
        inputs = Input( shape=input_size )
        concat_axis = 3 

        # replacing zhixuhao for zizhaozhang, to avoid different shapes of layers
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis) 
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        
        # ##Output Layer
        #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
        #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        

        model = Model(inputs=inputs, outputs=conv10)

        ##Compiling Model
        model.compile(optimizer = Adam(learning_rate = 1e-4), 
                      loss = 'binary_crossentropy', 
                      metrics = [MeanIoU(num_classes=2, name="meaniou"),
                                Precision(name="precision"),
                                Recall(name="recall"),
                                AUC(name="auc"),
                                BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                                self.jacard_coef]
                    )

                             

        model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

