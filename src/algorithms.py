import numpy as np
from abc import ABC, abstractmethod

class BeamformerSolver(ABC):
    """
    Abstract base class for beamforming solvers.
    """

    @abstractmethod
    def solve(self, A: np.ndarray, d: np.ndarray, current_weights: np.ndarray = None) -> np.ndarray:
        """
        Solves for weights w such that A^H w ~ d, subject to constraints.

        Args:
            A: Constraint Matrix (N_elements x N_constraints).
               Usually columns are steering vectors.
            d: Target response vector (N_constraints,).
               e.g. [1, 0, 0...] for [target, null1, null2...]
            current_weights: Optional initial weights for iterative solvers.

        Returns:
            Complex weights vector w (N_elements,).
        """
        pass

class ProjectionSolver(BeamformerSolver):
    """
    Standard Pseudo-Inverse Solver with simple Phase Projection.
    Fast but low quality (doesn't respect phase constraint during optimization).
    """

    def solve(self, A: np.ndarray, d: np.ndarray, current_weights: np.ndarray = None) -> np.ndarray:
        # 1. Unconstrained Least Squares: w = A(A^H A)^-1 d
        # Using pinv for stability: w = pinv(A^H) * d
        # A^H is (N_constraints, N_elements)
        # We want to solve A^H w = d
        # The minimum norm solution is w = A * (A^H A)^-1 d = pinv(A^H) * d

        # Note: numpy.linalg.pinv calculates pseudo-inverse of a matrix M.
        # If we have M x = b, then x = pinv(M) b.
        # Here our equation is A^H w = d.
        # So M = A^H.

        M = A.conj().T
        w_lin = np.linalg.pinv(M) @ d

        # 2. Phase Projection
        # Project onto unit circle
        # Handle zero magnitude to avoid nan
        eps = 1e-9
        w_phase = w_lin / (np.abs(w_lin) + eps)

        return w_phase

class IterativePOCSSolver(BeamformerSolver):
    """
    Alternating Projections (POCS) Solver.
    Iterates between:
    1. Linear Constraint Subspace (C1): {w | A^H w = d}
    2. Phase-Only Subspace (C2): {w | |w_i| = 1}

    Since intersection might be empty, this finds a solution close to both.
    """

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations

    def solve(self, A: np.ndarray, d: np.ndarray, current_weights: np.ndarray = None) -> np.ndarray:
        num_elements = A.shape[0]

        # Initialize weights
        if current_weights is not None:
            w = current_weights.copy()
        else:
            # Random initialization on unit circle
            phases = np.random.uniform(0, 2*np.pi, num_elements)
            w = np.exp(1j * phases)

        # Pre-compute Projection Operator for Linear Constraint
        # P_lin(w) = w - A(A^H A)^-1 (A^H w - d)
        # Let M = A^H.
        # We want to project w onto {x | M x = d}.
        # Formula: x_proj = x - M^H (M M^H)^-1 (M x - d)
        # Note: A is (N_elem, N_constr).
        # M = A^H is (N_constr, N_elem).
        # M M^H = A^H A is (N_constr, N_constr).
        # This is usually small (few nulls), so inversion is cheap.

        M = A.conj().T

        # Check singularity of M M^H
        # In beamforming, steering vectors are usually linearly independent unless angles are identical.
        # Using pinv handles rank deficiency gracefully.
        gram = M @ M.conj().T
        inv_gram = np.linalg.pinv(gram)
        projector = M.conj().T @ inv_gram

        for _ in range(self.max_iterations):
            # 1. Project onto Linear Constraints (Affine Subspace)
            # w_lin = w - A * inv(A^H A) * (A^H w - d)
            error = M @ w - d
            correction = projector @ error
            w_lin = w - correction

            # 2. Project onto Unit Circle (Phase Constraint)
            # w_{k+1} = exp(j * angle(w_lin))
            eps = 1e-9
            w = w_lin / (np.abs(w_lin) + eps)

        return w
