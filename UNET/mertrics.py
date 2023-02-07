
import keras.backend as K
import numpy as np

# from sklearn.metrics import sensitive_score, specificity_score
# specificity = specificity_score(y_test, y_pred)
# sensitive = sensitive_score(y_test, y_pred)
smooth=100
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# def specificity(y_true: np.array, y_pred: np.array, classes: set = None):

#     if classes is None: # Determine classes from the values
#         classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))

#     specs = []
#     for cls in classes:
#         y_true_cls = (y_true == cls).astype(int)
#         y_pred_cls = (y_pred == cls).astype(int)

#         fp = sum(y_pred_cls[y_true_cls != 1])
#         tn = sum(y_pred_cls[y_true_cls == 0] == False)

#         specificity_val = tn / (tn + fp)
#         specs.append(specificity_val)

#     return np.mean(specs)

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

ALPHA = 0.5
BETA = 0.5

def Tversky(y_true, y_pred, alpha=ALPHA, beta=BETA, smooth=100):
        
        #flatten label and prediction tensors
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((y_pred * y_true))
        FP = K.sum(((1-y_true) * y_pred))
        FN = K.sum((y_true * (1-y_pred)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return Tversky