import numpy as np


class SoftmaxRegression:
    """Softmax Regression đa lớp, huấn luyện bằng full-batch Gradient Descent."""

    def __init__(self, n_classes=10, lr=0.5, n_iters=300, verbose=False):
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        self.W = None  # (n_features, n_classes)
        self.b = None  # (n_classes,)
        self.loss_history = []

    @staticmethod
    def softmax(Z):
        # trừ max theo từng hàng để exp không bị tràn số
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(Y, P):
        eps = 1e-12  # tránh log(0)
        return -np.mean(np.sum(Y * np.log(np.clip(P, eps, 1.0)), axis=1))

    def _one_hot(self, y):
        Y = np.zeros((y.shape[0], self.n_classes))
        Y[np.arange(y.shape[0]), y] = 1.0
        return Y

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)
        self.loss_history = []
        Y = self._one_hot(y)

        for it in range(1, self.n_iters + 1):
            P = self.softmax(X @ self.W + self.b)
            self.loss_history.append(self.cross_entropy(Y, P))

            # dL/dz = (y_hat - y); gradient của W là x^T (y_hat - y), lấy trung bình trên N
            error = P - Y
            grad_W = X.T @ error / n_samples
            grad_b = error.mean(axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if self.verbose and (it == 1 or it % 50 == 0):
                print(f"[iter {it:>4}] loss = {self.loss_history[-1]:.6f}")
        return self

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
