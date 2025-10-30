import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def sample_prob(probs):
    """Convert probabilities into binary samples."""
    return (np.random.rand(*probs.shape) < probs).astype(np.float32)


class RBM:
    """Restricted Boltzmann Machine implemented from scratch using NumPy."""

    def __init__(self, n_visible, n_hidden, lr=0.1, seed=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr

        if seed is not None:
            np.random.seed(seed)

        # Xavier-style small random initialization
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.b = np.zeros(n_visible)  # visible bias
        self.c = np.zeros(n_hidden)   # hidden bias

    def v_to_h(self, v):
        """Compute hidden unit activations given visible units."""
        return sigmoid(np.dot(v, self.W) + self.c)

    def h_to_v(self, h):
        """Compute visible unit activations given hidden units."""
        return sigmoid(np.dot(h, self.W.T) + self.b)

    def contrastive_divergence(self, v0, k=1):
        """Perform k-step Contrastive Divergence update."""
        # Positive phase
        h0_prob = self.v_to_h(v0)
        h0_sample = sample_prob(h0_prob)

        v_k = v0
        h_k = h0_sample

        for _ in range(k):
            v_k_prob = self.h_to_v(h_k)
            v_k = sample_prob(v_k_prob)
            h_k_prob = self.v_to_h(v_k)
            h_k = sample_prob(h_k_prob)

        # Weight updates
        self.W += self.lr * (np.dot(v0.T, h0_prob) - np.dot(v_k.T, h_k_prob)) / v0.shape[0]
        self.b += self.lr * np.mean(v0 - v_k, axis=0)
        self.c += self.lr * np.mean(h0_prob - h_k_prob, axis=0)

        # Reconstruction error
        return np.mean((v0 - v_k_prob) ** 2)

    def transform(self, v):
        """Return hidden layer activations for given visible data."""
        return self.v_to_h(v)

    def reconstruct(self, v):
        """Reconstruct visible units from hidden representation."""
        h = self.v_to_h(v)
        v_recon = self.h_to_v(h)
        return v_recon
