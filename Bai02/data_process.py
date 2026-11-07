"""Đọc và tiền xử lý MNIST cho bài toán phân loại nhị phân (chữ số 0 và 1)."""
import gzip
import os
import struct

import numpy as np

MNIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mnist")

_FILES = {
    "x_train": "train-images-idx3-ubyte",
    "y_train": "train-labels-idx1-ubyte",
    "x_test": "t10k-images-idx3-ubyte",
    "y_test": "t10k-labels-idx1-ubyte",
}


def _open_idx(folder, name):
    """Mở file idx, hỗ trợ cả bản nén .gz lẫn bản thường."""
    plain = os.path.join(folder, name)
    if os.path.exists(plain):
        return open(plain, "rb")
    if os.path.exists(plain + ".gz"):
        return gzip.open(plain + ".gz", "rb")
    raise FileNotFoundError(f"Không tìm thấy {plain}(.gz)")


def _read_images(folder, name):
    with _open_idx(folder, name) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "File ảnh idx không hợp lệ"
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def _read_labels(folder, name):
    with _open_idx(folder, name) as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, "File nhãn idx không hợp lệ"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(folder=MNIST_DIR):
    """Trả về (x_train, y_train, x_test, y_test) với ảnh đã flatten thành 784 chiều."""
    return (
        _read_images(folder, _FILES["x_train"]),
        _read_labels(folder, _FILES["y_train"]),
        _read_images(folder, _FILES["x_test"]),
        _read_labels(folder, _FILES["y_test"]),
    )


def prepare_binary_mnist(folder=MNIST_DIR, digits=(0, 1)):
    """Giữ lại 2 chữ số, đưa pixel về [0, 1], nhãn: digits[0] -> 0, digits[1] -> 1."""
    x_tr, y_tr, x_te, y_te = load_mnist(folder)

    def _select(x, y):
        mask = np.isin(y, digits)
        x_sel = x[mask].astype(np.float64) / 255.0
        y_sel = (y[mask] == digits[1]).astype(np.float64)
        return x_sel, y_sel

    x_tr, y_tr = _select(x_tr, y_tr)
    x_te, y_te = _select(x_te, y_te)
    return x_tr, y_tr, x_te, y_te


def prepare_mnist_full(folder=MNIST_DIR):
    """Dùng đủ 10 chữ số: pixel về [0, 1], nhãn là số nguyên 0..9."""
    x_tr, y_tr, x_te, y_te = load_mnist(folder)
    return (
        x_tr.astype(np.float64) / 255.0,
        y_tr.astype(np.int64),
        x_te.astype(np.float64) / 255.0,
        y_te.astype(np.int64),
    )
