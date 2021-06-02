import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K 
import tensorflow as tf

from tensorflow.keras.metrics import *

class CustomMetrics:
    
    @staticmethod
    def get_recall(tp, fn):
        recall = tp / (tp + fn)
        print(f"The recall value is: {recall}")
        return recall

    @staticmethod
    def get_precision(tp, tn):
        precision = tp / (tp + tp)
        print(f"The precision value is: {precision}")
        return precision

    @staticmethod
    def get_fscore(recall, precision):
        f_score = (2 * recall * precision) / (recall + precision)
        print(f"The f_score value is: {f_score}")
        return f_score

    @staticmethod
    def get_fbetaScore(beta, precision, recall):

        # F0.5-Measure  (beta=0.5): More weight on precision, less weight on recall
        # F1-Measure    (beta=1.0): Balance the weight on precision and recall
        # F2-Measure    (beta=2.0): Less weight on precision, more weight on recall
        f_beta_score = ((1 + beta ** 2) * precision * recall) / (
            beta ** 2 * precision + recall
        )
        print(f"The f_beta_score value is: {f_beta_score}")
        return f_beta_score

    @staticmethod
    def plot_roc_curve(fpr, tpr):
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()

    @staticmethod
    def get_auc(model, testX, testY):
        probs = model.predict_proba(testX)
        probs = probs[:, 1]

        auc = roc_auc_score(testY, probs)
        print("AUC: %.2f" % auc)

        fpr, tpr, thresholds = roc_curve(testY, probs)
        CustomMetrics.plot_roc_curve(fpr, tpr)

        return auc

    # Jaccard Index
    @staticmethod
    def iou_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
    
    @staticmethod
    def iou(y_true, y_pred, label: int):
        """
        Return the Intersection over Union (IoU) for a given label.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
            label: the label to return the IoU for
        Returns:
            the IoU for the given label
        """
        # extract the label values using the argmax operator then
        # calculate equality of the predictions and truths to the label
        y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
        y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        # avoid divide by zero - if the union is zero, return 1
        # otherwise, return the intersection over union
        return K.switch(K.equal(union, 0), 1.0, intersection / union)

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    @staticmethod
    def jacard_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

    @staticmethod
    def jacard_coef_loss(y_true, y_pred):
        return -CustomMetrics.jacard_coef(y_true, y_pred)


    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.dot(y_true, K.transpose(y_pred))
        union = K.dot(y_true,K.transpose(y_true))+K.dot(y_pred,K.transpose(y_pred))
        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    def dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    #https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    @staticmethod
    def dice_coef2(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice


    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return K.mean(1-CustomMetrics.dice_coef(y_true, y_pred),axis=-1)


    @staticmethod
    def dice_coefficient(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return numerator / (denominator + tf.keras.backend.epsilon())
 