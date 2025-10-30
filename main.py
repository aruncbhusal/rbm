import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model.rbm import RBM


def load_mnist(n_samples=10000):
    """Load MNIST dataset and preprocess."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="pandas")
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=42)
    return X.values if hasattr(X, "values") else X, y


def train_rbm(X_train, n_hidden=128, lr=0.1, epochs=5, batch_size=64):
    rbm = RBM(n_visible=X_train.shape[1], n_hidden=n_hidden, lr=lr, seed=42)

    print(f"Training RBM with {n_hidden} hidden units for {epochs} epochs...")
    for epoch in range(epochs):
        np.random.shuffle(X_train)
        losses = []
        for i in range(0, X_train.shape[0], batch_size):
            batch = X_train[i:i + batch_size]
            loss = rbm.contrastive_divergence(batch)
            losses.append(loss)

        print(f"Epoch {epoch + 1}/{epochs}, Reconstruction Error: {np.mean(losses):.6f}")
    return rbm


def evaluate_with_logreg(rbm, X_train, y_train):
    """Use hidden activations as features for classification."""
    print("Extracting features using trained RBM...")
    H_train = rbm.transform(X_train)

    print("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=500)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        H_train, y_train, test_size=0.2, random_state=42
    )
    clf.fit(X_train_split, y_train_split)

    y_pred = clf.predict(X_test_split)
    acc = accuracy_score(y_test_split, y_pred)
    print(f"Classification accuracy using RBM features: {acc * 100:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Restricted Boltzmann Machine Trainer")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--hidden", type=int, default=128, help="Number of hidden units")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    X_train, y_train = load_mnist(n_samples=args.samples)
    rbm = train_rbm(X_train, n_hidden=args.hidden, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    evaluate_with_logreg(rbm, X_train, y_train)


if __name__ == "__main__":
    main()
