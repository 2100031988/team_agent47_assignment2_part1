import numpy as np


# =========================
# ACCURACY
# =========================
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


# =========================
# CONFUSION MATRIX
# =========================
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


# =========================
# PRECISION
# =========================
def precision(y_true, y_pred, average="binary"):
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp

    precision_vals = tp / (tp + fp + 1e-12)

    if average == "binary":
        return precision_vals.mean()
    elif average == "macro":
        return np.mean(precision_vals)
    else:
        raise ValueError("Invalid average method")


# =========================
# RECALL
# =========================
def recall(y_true, y_pred, average="binary"):
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp

    recall_vals = tp / (tp + fn + 1e-12)

    if average == "binary":
        return recall_vals.mean()
    elif average == "macro":
        return np.mean(recall_vals)
    else:
        raise ValueError("Invalid average method")


# =========================
# F1 SCORE
# =========================
def f1_score(y_true, y_pred, average="binary"):
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / (p + r + 1e-12)


# =========================
# MEAN SQUARED ERROR
# =========================
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


# =========================
# ROC CURVE (binary only)
# =========================
def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = np.sort(np.unique(y_score))[::-1]

    tpr_list = []
    fpr_list = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr = tp / (P + 1e-12)
        fpr = fp / (N + 1e-12)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)


# =========================
# AUC (Trapezoidal rule)
# =========================
def auc(fpr, tpr):
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    return np.trapz(tpr, fpr)