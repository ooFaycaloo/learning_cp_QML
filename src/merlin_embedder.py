import numpy as np
from utils import get_logger

LOGGER = get_logger(__name__)

class MerlinEmbedder:
    """
    Wrapper "safe" : essaye d'utiliser MerLin si dispo, sinon fallback PCA-like identity.
    Le but du hackathon: intégrer une brique QML **sans casser** le pipeline.
    """
    def __init__(self, n_qubits: int = 4, n_features_in: int = 4):
        self.n_qubits = n_qubits
        self.n_features_in = n_features_in
        self._backend = None

    def fit(self, X: np.ndarray, y=None):
        try:
            import merlin  # noqa: F401
            # TODO: brancher l'API MerLin exacte (dépend de la version fournie au hackathon)
            self._backend = "merlin"
            LOGGER.info("MerLin detected: using QML backend placeholder (to be connected to actual MerLin API).")
        except Exception as e:
            self._backend = None
            LOGGER.warning("MerLin not available in this environment. Falling back to identity embedder. Err=%s", e)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "merlin":
            # PLACEHOLDER: ici vous mettez l'appel MerLin qui retourne un embedding.
            # En attendant: on applique une non-linéarité simple pour simuler un embedding.
            return np.tanh(X)
        return X

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
