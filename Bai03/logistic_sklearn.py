"""Bài 3a: Logistic Regression bằng scikit-learn - MNIST chữ số 0 và 1."""
from sklearn.linear_model import LogisticRegression

from data_process import prepare_binary_mnist
from report import print_report


def main():
    x_train, y_train, x_test, y_test = prepare_binary_mnist(digits=(0, 1))
    print(f"Train: {x_train.shape}, Test: {x_test.shape}\n")

    model = LogisticRegression(solver="lbfgs", max_iter=300)
    model.fit(x_train, y_train)

    print_report(
        y_train, model.predict(x_train), y_test, model.predict(x_test),
        average="binary", target_names=["0", "1"],
    )


if __name__ == "__main__":
    main()
