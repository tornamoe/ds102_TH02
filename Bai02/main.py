import numpy as np

from data_process import prepare_mnist_full
from metrics import multiclass_report, confusion_matrix
from SoftmaxRegression import SoftmaxRegression
from visual import plot_loss


def main():
    x_train, y_train, x_test, y_test = prepare_mnist_full()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = SoftmaxRegression(n_classes=10, lr=0.5, n_iters=300, verbose=True)
    model.fit(x_train, y_train)

    plot_loss(model.loss_history, "Softmax Regression - MNIST 10 lớp", "loss_curve_softmax.png")

    y_pred = model.predict(x_test)
    acc, per_class = multiclass_report(y_test, y_pred, 10)

    print("\n=== Kết quả trên tập test ===")
    print(f"{'Lớp':<6}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    for c in range(10):
        p, r, f = per_class[c]
        print(f"{c:<6}{p:>12.4f}{r:>10.4f}{f:>10.4f}")
    macro = per_class.mean(axis=0)
    print(f"{'Macro':<6}{macro[0]:>12.4f}{macro[1]:>10.4f}{macro[2]:>10.4f}")
    print(f"\nAccuracy: {acc:.4f}")

    print("\nMa trận nhầm lẫn (hàng: thực tế, cột: dự đoán):")
    print(confusion_matrix(y_test, y_pred, 10))


if __name__ == "__main__":
    main()
