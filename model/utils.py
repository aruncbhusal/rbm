import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist(n_samples=10000, binarize=False, seed=42):
    """
    Load MNIST dataset, normalize to [0,1], and optionally binarize.
    Returns: X_train, y_train, X_test, y_test
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="pandas")
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # Split and optionally binarize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_samples, stratify=y, random_state=seed
    )

    if binarize:
        X_train = (X_train > 0.5).astype(np.float32)
        X_test = (X_test > 0.5).astype(np.float32)

    # Return as np arrays if they’re pandas objects
    if hasattr(X_train, "values"):
        X_train, X_test = X_train.values, X_test.values

    return X_train, y_train, X_test, y_test


def visualize_reconstruction(original, reconstructed, n=10):
    """Show n original and reconstructed images side by side."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(n, 2))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Original")

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Reconstructed")

    plt.tight_layout()
    plt.show()
