import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K 

class CustomMetrics:
    
    @staticmethod
    def get_recall(self, tp, fn):
        recall = tp / (tp + fn)
        print(f"The recall value is: {recall}")
        return recall

    @staticmethod
    def get_precision(self, tp, tn):
        precision = tp / (tp + tp)
        print(f"The precision value is: {precision}")
        return precision

    @staticmethod
    def get_f1_measure(self, y_true, y_pred):
        precision = self.get_precision(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    @staticmethod
    def get_fscore(self, recall, precision):
        f_score = (2 * recall * precision) / (recall + precision)
        print(f"The f_score value is: {f_score}")
        return f_score

    @staticmethod
    def get_fbetaScore(self, beta, precision, recall):

        # F0.5-Measure  (beta=0.5): More weight on precision, less weight on recall
        # F1-Measure    (beta=1.0): Balance the weight on precision and recall
        # F2-Measure    (beta=2.0): Less weight on precision, more weight on recall
        f_beta_score = ((1 + beta ** 2) * precision * recall) / (
            beta ** 2 * precision + recall
        )
        print(f"The f_beta_score value is: {f_beta_score}")
        return f_beta_score

    @staticmethod
    def plot_roc_curve(self, fpr, tpr):
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()

    @staticmethod
    def get_auc(self, model, testX, testY):
        probs = model.predict_proba(testX)
        probs = probs[:, 1]

        auc = roc_auc_score(testY, probs)
        print("AUC: %.2f" % auc)

        fpr, tpr, thresholds = roc_curve(testY, probs)
        self.plot_roc_curve(fpr, tpr)

        return auc

    # Jaccard Index
    @staticmethod
    def iou_coef(self, y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    @staticmethod
    def jacard_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

    @staticmethod
    def jacard_coef_loss(self, y_true, y_pred):
        return -self.jacard_coef(y_true, y_pred)