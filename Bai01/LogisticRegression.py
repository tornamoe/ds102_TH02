import numpy as np


class LogisticRegression:
    """Logistic Regression nhị phân, huấn luyện bằng full-batch Gradient Descent."""

    def __init__(self, lr=0.1, n_iters=300, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        # clip để exp không bị tràn số
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def cross_entropy(y, y_hat):
        eps = 1e-12  # tránh log(0)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for it in range(1, self.n_iters + 1):
            y_hat = self.sigmoid(X @ self.w + self.b)
            self.loss_history.append(self.cross_entropy(y, y_hat))

            # dL/dw = (y_hat - y) x, lấy trung bình trên N mẫu
            error = y_hat - y
            grad_w = X.T @ error / n_samples
            grad_b = error.mean()

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if self.verbose and (it == 1 or it % 50 == 0):
                print(f"[iter {it:>4}] loss = {self.loss_history[-1]:.6f}")
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
