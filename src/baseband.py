import numpy as np
from typing import List, Tuple, Dict

class BasebandProcessor:
    """
    Central Baseband Processor.
    Optimizes the combination of synthesized beams from multiple RF chains.
    """

    def __init__(self):
        self.weights = None

    def optimize_weights(
        self,
        rf_chains: List,
        target: Tuple[float, float],
        interferers: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Calculates optimal baseband weights to combine RF signals.
        Goal: Gain=1 at Target, Gain=0 at Interferers/Risk Angles.
        Method: Minimum Norm Solution (Pseudo-Inverse).

        Args:
            rf_chains: List of RFChain objects.
            target: (az, el) tuple.
            interferers: List of (az, el) tuples (Explicit Nulls + Risk Angles).

        Returns:
            Complex weights vector (size M).
        """
        num_chains = len(rf_chains)
        if num_chains == 0:
            return np.array([])

        # 1. Build Response Matrix H (K_constraints x M_chains)
        # Constraints:
        # Row 0: Target response = 1
        # Row 1..K: Interferer response = 0

        constraints_dirs = [target] + interferers
        # Scale target constraint to number of chains to represent array gain
        constraints_vals = [float(num_chains) + 0j] + [0.0 + 0j] * len(interferers)

        H_rows = []

        for az, el in constraints_dirs:
            # For each direction, get the complex response of every RF chain
            # row = [AF_1(theta), AF_2(theta), ..., AF_M(theta)]
            row_responses = []
            for chain in rf_chains:
                resp = chain.get_complex_response(az, el)
                row_responses.append(resp)
            H_rows.append(row_responses)

        H = np.array(H_rows, dtype=complex) # (K, M)
        d = np.array(constraints_vals, dtype=complex) # (K,)

        # 2. Solve H w = d
        # If K < M (Underdetermined): Many solutions, pick min norm (pinv).
        # If K > M (Overdetermined): No exact solution, pick min error (pinv).
        # pinv handles both.

        self.weights = np.linalg.pinv(H) @ d

        return self.weights

    def process_signals(self, rf_signals: List[complex]) -> complex:
        """
        Apply computed weights to incoming signals.
        """
        if self.weights is None:
            raise ValueError("Weights not optimized yet. Call optimize_weights first.")
        if len(rf_signals) != len(self.weights):
            raise ValueError("Signal dimension mismatch.")

        return np.dot(self.weights, np.array(rf_signals))
