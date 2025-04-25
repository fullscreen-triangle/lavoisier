import numpy as np

def gauss(x: np.ndarray, mu: float, sigma: float, amp: float):
    """gaussian curve."""
    gaussian = amp * np.power(np.e, - 0.5 * ((x - mu) / sigma) ** 2)
    return gaussian

def gaussian_mixture(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Mixture of gaussian curves."""
    mixture = np.zeros((params.shape[0], x.size))
    for k_row, param in enumerate(params):
        mixture[k_row] = gauss(x, *param)
    return mixture
