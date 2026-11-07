"""Bài 3b: Softmax Regression bằng scikit-learn - MNIST 10 chữ số."""
from sklearn.linear_model import LogisticRegression

from data_process import prepare_mnist_full
from report import print_report


def main():
    x_train, y_train, x_test, y_test = prepare_mnist_full()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}\n")

    # Với nhiều hơn 2 lớp, lbfgs tối ưu hàm mất mát softmax (multinomial) trực tiếp
    model = LogisticRegression(solver="lbfgs", max_iter=300)
    model.fit(x_train, y_train)

    print_report(
        y_train, model.predict(x_train), y_test, model.predict(x_test),
        average="macro", target_names=[str(c) for c in range(10)],
    )


if __name__ == "__main__":
    main()
