"""
Merlin (merlinquantum) -> sklearn-style embedder.

PyPI package name: merlinquantum
Python import name: merlin

This transformer produces a fixed-size embedding from input features using
MerLin's QuantumLayer + a lightweight grouping head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from utils import get_logger

logger = get_logger("merlin_embedder")


_IMPORT_ERR: Optional[Exception] = None
HAS_MERLIN = False

try:
    import torch
    from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
    from merlin.builder import CircuitBuilder
    from merlin import ComputationSpace  # optional, may not exist in older versions

    HAS_MERLIN = True
except Exception as e:  # pragma: no cover
    _IMPORT_ERR = e
    HAS_MERLIN = False


@dataclass
class _MerlinConfig:
    n_qubits: int = 4
    n_features_in: int = 4
    n_layers: int = 1
    shots: int = 0  # 0 = analytic/simulator default; kept for API compatibility
    seed: int = 0


class MerlinEmbedder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that maps X (n_samples, n_features_in) -> Z (n_samples, n_qubits).

    Notes
    -----
    - This embedder is intentionally light: it does not train the quantum layer; it only
      performs a forward pass to build an embedding (feature map).
    - If you want a trained QML model, you would train the QuantumLayer (torch) end-to-end.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_features_in: int = 4,
        n_layers: int = 1,
        shots: int = 0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if not HAS_MERLIN:
            raise ImportError(
                "MerLin (pip: merlinquantum, import: merlin) is not available in this environment."
            ) from _IMPORT_ERR

        self.cfg = _MerlinConfig(
            n_qubits=int(n_qubits),
            n_features_in=int(n_features_in),
            n_layers=int(n_layers),
            shots=int(shots),
            seed=int(seed),
        )
        self.device = device

        # Will be created in fit()
        self._model = None

    def _build_model(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # MerLin uses photonic modes; "dual rail" often uses 2 modes per qubit.
        # We keep it simple and allocate 2*n_qubits modes.
        n_modes = 2 * self.cfg.n_qubits

        builder = CircuitBuilder(n_modes=n_modes)

        # Add a few entangling/rotation blocks
        for i in range(self.cfg.n_layers):
            builder.add_entangling_layer(trainable=True, name=f"U{i+1}")
            # Angle encode as many features as available; map onto first modes.
            m = min(self.cfg.n_features_in, n_modes)
            builder.add_angle_encoding(
                modes=list(range(m)),
                name=f"input{i+1}",
                scale=np.pi,
            )
            builder.add_rotations(trainable=True, name=f"theta{i+1}")
            builder.add_superpositions(depth=1, trainable=True)

        # QuantumLayer forward returns classical tensor according to measurement_strategy.
        # PROBABILITIES tends to be the most robust output for downstream sklearn models.
        kwargs = dict(
            input_size=self.cfg.n_features_in,
            builder=builder,
            n_photons=self.cfg.n_qubits,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        # ComputationSpace may not exist in all versions; if it does, dual rail is sensible.
        try:
            kwargs["computation_space"] = ComputationSpace.DUAL_RAIL  # type: ignore[name-defined]
        except Exception:
            pass

        qlayer = QuantumLayer(**kwargs)

        # Group the (often large) output space down to n_qubits features.
        head = torch.nn.Sequential(
            qlayer,
            LexGrouping(qlayer.output_size, self.cfg.n_qubits),
        )

        return head

    def fit(self, X, y=None):
        X = self._validate_X(X)
        # Ensure we keep consistent feature count
        self.cfg.n_features_in = int(X.shape[1])

        self._model = self._build_model()
        self._model.to(self.device)
        self._model.eval()

        logger.info(
            "Initialized MerLin embedder",
            extra={
                "n_qubits": self.cfg.n_qubits,
                "n_features_in": self.cfg.n_features_in,
                "n_layers": self.cfg.n_layers,
                "device": self.device,
            },
        )
        return self

    def transform(self, X):
        if self._model is None:
            raise RuntimeError("MerlinEmbedder is not fitted. Call fit() first.")
        X = self._validate_X(X)

        import torch

        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            z = self._model(xt)
            # Ensure numpy output
            z = z.detach().cpu().numpy()

        return z

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape={X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf.")
        return X
