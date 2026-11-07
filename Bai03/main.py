import logistic_sklearn
import softmax_sklearn

if __name__ == "__main__":
    print("=" * 50)
    print("Logistic Regression (sklearn) - MNIST 0 vs 1")
    print("=" * 50)
    logistic_sklearn.main()

    print("\n" + "=" * 50)
    print("Softmax Regression (sklearn) - MNIST 10 lớp")
    print("=" * 50)
    softmax_sklearn.main()
