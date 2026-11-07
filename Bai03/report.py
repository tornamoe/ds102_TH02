"""Hàm dùng chung để in kết quả đánh giá cho Bài 3."""
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def print_report(y_train, y_train_pred, y_test, y_test_pred, average, target_names=None):
    """In bảng Train/Test (precision, recall, F1, accuracy) và classification report trên tập test."""
    tag = "" if average == "binary" else "Macro "
    rows = [
        (f"{tag}Precision", precision_score, {"average": average}),
        (f"{tag}Recall", recall_score, {"average": average}),
        (f"{tag}F1-Score", f1_score, {"average": average}),
    ]

    print(f"{'Metric':<18} | {'Train':>10} | {'Test':>10}")
    print("-" * 44)
    for name, fn, kw in rows:
        print(f"{name:<18} | {fn(y_train, y_train_pred, **kw):>10.4f} | {fn(y_test, y_test_pred, **kw):>10.4f}")
    print(f"{'Accuracy':<18} | {accuracy_score(y_train, y_train_pred):>10.4f} | {accuracy_score(y_test, y_test_pred):>10.4f}")

    print("\nClassification report (tập test):")
    print(classification_report(y_test, y_test_pred, target_names=target_names, digits=4))
