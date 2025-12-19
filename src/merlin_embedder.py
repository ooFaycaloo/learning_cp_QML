import numpy as np
from utils import get_logger

LOGGER = get_logger(__name__)

try:
    from merlinquantum import Merlin
    HAS_MERLIN = True
except ImportError as e:
    HAS_MERLIN = False
    _IMPORT_ERR = e


class MerlinEmbedder:
    """
    Real MerLin (Quandela) quantum feature map wrapper.
    """

    def __init__(
        self,
        n_qubits: int,
        n_features_in: int,
        n_layers: int = 2,
        shots: int = 1024,
        seed: int = 42,
    ):
        if not HAS_MERLIN:
            raise ImportError(
                "MerLin official (merlinquantum) not installed"
            ) from _IMPORT_ERR

        self.n_qubits = n_qubits
        self.n_features_in = n_features_in
        self.n_layers = n_layers
        self.shots = shots
        self.seed = seed

        self.model = Merlin(
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
            seed=seed,
        )

        LOGGER.info(
            "MerLin OFFICIAL initialized | qubits=%d layers=%d shots=%d",
            n_qubits, n_layers, shots
        )

    def fit(self, X: np.ndarray, y=None):
        X = self._check(X)
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check(X)
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = self._check(X)
        return self.model.fit_transform(X)

    def _check(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Expected {self.n_features_in} features, got {X.shape[1]}"
            )
        return X
