import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import *  
from tensorflow.keras import backend as K
from tensorflow.python.keras import losses, regularizers
from tensorflow.python.keras.backend import backend
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
        output_layer = Conv2D(3, 3, activation = tfa.activations.sparsemax, padding = 'same')(conv9)

        ##Defining Model
        model = Model(inputs, output_layer)

        ##Compiling Model 
        model.compile(optimizer = Adam(learning_rate = 1e-4), 
                    loss = tfa.losses.SparsemaxLoss(from_logits = True,
                                                    reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                                                    name = 'sparsemax_loss'),
                    metrics = [MeanIoU(num_classes=2, name="mean_iou"),
                                Accuracy(name="accuracy"),
                                Precision(name="precision"),
                                Recall(name="recall"),
                                AUC(name="auc"),
                                CustomMetrics.jacard_coef,
                                CustomMetrics.dice_coefficient,
                                tfa.metrics.F1Score(
                                                        num_classes = 2,
                                                        average = 'micro',
                                                        name = 'f1_score'
                                                    )]
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

    def create_model_keras(self, img_size, num_classes):
        inputs = Input(shape=img_size + (3,))

        ### [First half of the network: downsampling inputs] ###
        #2021-05-31 18:27:21.499535: W tensorflow/core/framework/op_kernel.cc:1767] OP_REQUIRES failed at conv_ops_fused_impl.h:769 : 
        # Invalid argument: input depth must be evenly divisible by filter depth: 3 vs 2
        ##Incompatible shapes: [4,416,320,2] vs. [4,416,320,3]
	    #[[node gradient_tape/binary_crossentropy/logistic_loss/mul/BroadcastGradientArgs (defined at /Users/charles/GoogleDrive/Mestrado/_orientacao/codes/unet/unet_helper.py:344) ]] [Op:__inference_train_function_6536]


        # Entry block
        x = Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = UpSampling2D(2)(x)

            # Project residual
            residual = UpSampling2D(2)(previous_block_activation)
            residual = Conv2D(filters, 1, padding="same")(residual)
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = Conv2D(3, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = Model(inputs, outputs)
        model.compile(optimizer = Adam(learning_rate = 1e-4), 
                      loss = tfa.losses.GIoULoss(
                              mode = 'iou',
                              reduction = tf.keras.losses.Reduction.AUTO,
                              name = 'iou_loss'
                            ),
                            # 'binary_crossentropy',
                      metrics = [MeanIoU(num_classes=2, name="mean_iou"),
                                 Accuracy(name="accuracy"),
                                 Precision(name="precision"),
                                 Recall(name="recall"),
                                 AUC(name="auc"),
                                 CustomMetrics.jacard_coef,
                                 CustomMetrics.dice_coefficient]
                    )
                             

        model.summary()

        return model


        # Free up RAM in case the model definition cells were run multiple times
        tf.keras.backend.clear_session()
        # Total params: 31,032,837