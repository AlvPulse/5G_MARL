import numpy as np
from typing import List, Tuple, Dict
from src.system import MultiPanelSystem

class SystemEvaluator:
    """
    Evaluates the performance of the beamforming system against Ground Truth.
    Calculates SINR for each Ground Truth target based on the beam generated for its hypothesis.
    """

    def __init__(self, system: MultiPanelSystem):
        self.system = system

    def compute_sinr_metrics(
        self,
        ground_truth_targets: List[Tuple[float, float]],
        beam_weights_dict: Dict[int, np.ndarray], # index -> weights
        hypothesized_targets: List[Tuple[float, float]],
        noise_power_db: float = -90.0
    ) -> Dict:
        """
        Computes SINR for each Ground Truth target.
        Assumes a 1-to-1 mapping between Ground Truth and Hypothesized targets
        (based on order or closest match).

        Args:
            ground_truth_targets: List of (az, el) real locations.
            beam_weights_dict: Weights generated for each hypothesized target stream.
            hypothesized_targets: The targets used to generate the beams.
            noise_power_db: Thermal noise power in dB (relative to unity).

        Returns:
            Dictionary mapping target index to evaluation metrics (SINR, Match Error).
        """
        results = {}
        noise_power_linear = 10**(noise_power_db / 10.0)

        # Determine Mapping: GT index -> Hypothesis index
        # We assume they are ordered correspondingly or find closest.
        # Let's find closest hypothesis for each GT.

        for gt_idx, gt_target in enumerate(ground_truth_targets):
            best_hyp_idx = -1
            min_dist = float('inf')

            for hyp_idx, hyp_target in enumerate(hypothesized_targets):
                # Simple Euclidean distance in (Az, El) space (approximate)
                # Or great circle. Let's use Euclidean for speed as angles are close.
                d = np.sqrt((gt_target[0] - hyp_target[0])**2 + (gt_target[1] - hyp_target[1])**2)
                if d < min_dist:
                    min_dist = d
                    best_hyp_idx = hyp_idx

            if best_hyp_idx == -1 or best_hyp_idx not in beam_weights_dict:
                results[gt_idx] = {"error": "No matching hypothesis found"}
                continue

            weights_signal = beam_weights_dict[best_hyp_idx]

            # 1. Calculate Signal Power (S)
            # Gain of Beam[best_hyp_idx] at GT location
            s_resp = self._get_system_response(gt_target, weights_signal)
            s_power = np.abs(s_resp)**2

            # 2. Calculate Interference Power (I)
            # Sum of Gains of ALL OTHER Beams at GT location
            i_power = 0.0
            for hyp_idx, weights_interf in beam_weights_dict.items():
                if hyp_idx == best_hyp_idx:
                    continue

                i_resp = self._get_system_response(gt_target, weights_interf)
                i_power += np.abs(i_resp)**2

            # 3. Compute SINR
            total_interference_noise = i_power + noise_power_linear
            if total_interference_noise < 1e-12:
                sinr_linear = float('inf')
            else:
                sinr_linear = s_power / total_interference_noise

            sinr_db = 10 * np.log10(sinr_linear + 1e-12)

            results[gt_idx] = {
                "matched_hypothesis_idx": best_hyp_idx,
                "hypothesis_error_deg": float(min_dist),
                "signal_power_db": 10 * np.log10(s_power + 1e-12),
                "interference_power_db": 10 * np.log10(i_power + 1e-12),
                "sinr_db": float(sinr_db)
            }

        return results

    def _get_system_response(self, target: Tuple[float, float], weights: np.ndarray) -> complex:
        """
        Computes composite system response for a specific weight vector at a target.
        """
        az, el = target
        total_resp = 0j

        # Re-use logic from system.analyze_system_performance but for single point
        for i, chain in enumerate(self.system.rf_chains):
            # Element Pattern
            pat = chain.panel.get_element_pattern(az, el)
            # Array Factor
            af = chain.get_complex_response(az, el)

            # Combined
            total_resp += weights[i] * af * pat

        return total_resp
