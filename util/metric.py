import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    if len(np.unique(y_true)) == 2:  # Binary classification
        auroc = roc_auc_score(y_true, y_pred)
        y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        return auroc, f1, acc, precision, recall
    
    else:  # Multi-class classification
        auroc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        y_pred = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        return auroc, f1, acc, precision, recall
