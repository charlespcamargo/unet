import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K 
import tensorflow as tf
from tensorflow.keras.metrics import *

import numpy as np
from scipy.ndimage import distance_transform_edt as distance

class CustomMetricsAndLosses:
    
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
    def get_fbeta_score(beta, precision, recall):

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
    def get_auc(model, test_x, test_y):
        probs = model.predict_proba(test_x)
        probs = probs[:, 1]

        auc = roc_auc_score(test_y, probs)
        print("AUC: %.2f" % auc)

        fpr, tpr, thresholds = roc_curve(test_y, probs)
        CustomMetricsAndLosses.plot_roc_curve(fpr, tpr)

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

    @staticmethod
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
        return -CustomMetricsAndLosses.jacard_coef(y_true, y_pred)


    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    @staticmethod
    def euclidean_distance_loss(y_true, y_pred):
        y = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

        return y

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        # y_true_f = K.flatten(y_true)
        # y_pred_f = K.flatten(y_pred)
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
        return K.mean(1-CustomMetricsAndLosses.dice_coef(y_true, y_pred),axis=-1)


    @staticmethod
    def dice_coefficient(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return numerator / (denominator + tf.keras.backend.epsilon())
    
    @staticmethod
    def calc_dist_map(seg):
        res = np.zeros_like(seg)
        posmask = seg.astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

        return res

    @staticmethod
    def calc_dist_map_batch(y_true):
        y_true_numpy = y_true.numpy()
        return np.array([CustomMetricsAndLosses.calc_dist_map(y)
                        for y in y_true_numpy]).astype(np.float32)

    @staticmethod
    def surface_loss(y_true, y_pred):
        y_true_dist_map = tf.py_function(func=CustomMetricsAndLosses.calc_dist_map_batch,
                                        inp=[y_true],
                                        Tout=tf.float32)
        multipled = y_pred * y_true_dist_map
        return K.mean(multipled)


    # weight: weighted tensor(same shape with mask image)
    @staticmethod
    def weighted_bce_loss(y_true, y_pred, weight):
        # avoiding overflow
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        logit_y_pred = K.log(y_pred / (1. - y_pred))
        
        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
        (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
        return K.sum(loss) / K.sum(weight)

    @staticmethod
    def weighted_dice_loss(y_true, y_pred, weight):
        smooth = 1.
        w, m1, m2 = weight * weight, y_true, y_pred
        intersection = (m1 * m2)
        score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
        loss = 1. - K.sum(score)
        return loss

    @staticmethod
    def weighted_bce_dice_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        # if we want to get same size of output, kernel size must be odd number
        averaged_mask = K.pool2d(
                y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
        border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
        weight = K.ones_like(averaged_mask)
        w0 = K.sum(weight)
        weight += border * 2
        w1 = K.sum(weight)
        weight *= (w0 / w1)
        loss = CustomMetricsAndLosses.weighted_bce_loss(y_true, y_pred, weight) + CustomMetricsAndLosses.weighted_dice_loss(y_true, y_pred, weight)
        
        return loss