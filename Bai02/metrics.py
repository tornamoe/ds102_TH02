import numpy as np


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def binary_report(y_true, y_pred, positive=1):
    """Accuracy, precision, recall, F1 của lớp dương."""
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    return (np.mean(y_true == y_pred), *_prf(tp, fp, fn))


def multiclass_report(y_true, y_pred, n_classes):
    """Trả về accuracy và bảng precision/recall/F1 theo từng lớp (n_classes x 3)."""
    per_class = np.zeros((n_classes, 3))
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        per_class[c] = _prf(tp, fp, fn)
    return np.mean(y_true == y_pred), per_class


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm
