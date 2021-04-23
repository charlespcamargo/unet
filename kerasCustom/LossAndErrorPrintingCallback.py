from tensorflow.keras.callbacks import Callback  

class LossAndErrorPrintingCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print("The average loss for epoch {} is {:7.2f} "
              "and mean absolute error is {:7.2f}.".format(epoch, logs["loss"], logs["mean_absolute_error"]))

