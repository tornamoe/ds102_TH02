import numpy as np

from data_process import prepare_binary_mnist
from LogisticRegression import LogisticRegression
from metrics import binary_report
from visual import plot_loss


def main():
    # Thư mục chứa 4 file MNIST dạng idx (có thể là .gz)
    x_train, y_train, x_test, y_test = prepare_binary_mnist(digits=(0, 1))
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = LogisticRegression(lr=0.1, n_iters=300, verbose=True)
    model.fit(x_train, y_train)

    plot_loss(model.loss_history, "Logistic Regression - MNIST 0 vs 1", "loss_curve.png")

    y_pred = model.predict(x_test)
    acc, prec, rec, f1 = binary_report(y_test.astype(int), y_pred)
    print("\n=== Kết quả trên tập test ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()
