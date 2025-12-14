import numpy as np
from typing import List, Tuple, Dict

from src.system import MultiPanelSystem
from src.baseband import BasebandProcessor

class BeamformingBrain:
    """
    Higher-level module for managing beamforming tasks and resource distribution.
    """

    def __init__(self, system: MultiPanelSystem):
        self.system = system

    def distribute_and_process(self, hypothesized_targets: List[Tuple[float, float]], fov_deg: float = 60.0) -> Dict:
        """
        Distributes targets to panels based on FoV, configures RF chains,
        and runs Baseband optimization for each target stream.

        Args:
            hypothesized_targets: List of (az, el) tuples.
            fov_deg: Field of View limit for panels.

        Returns:
            Dictionary mapping target index to its performance metrics.
        """
        if not hypothesized_targets:
            return {}

        # 1. RF Allocation (Resource Distribution)
        # Strategy: Each panel picks the "best" target in its FoV to serve as Primary.
        # All other visible targets become Nulls.

        for chain in self.system.rf_chains:
            panel = chain.panel
            visible_targets = []
            visible_indices = []

            # Find visible targets
            for i, tgt in enumerate(hypothesized_targets):
                if panel.is_visible(tgt[0], tgt[1], fov_deg=fov_deg):
                    visible_targets.append(tgt)
                    visible_indices.append(i)

            if not visible_targets:
                # Idle or Keep previous?
                # For now, steer to (0,0) or zenith?
                # Let's just hold current weights (do nothing) or steer to first hypothesized target weakly?
                # Better: Steer to a default "Scan" position?
                # Let's steer to (0, 90) [Zenith] as default safe state.
                chain.set_beam(0, 90, [])
                continue

            # Select Primary Target
            # Heuristic: Closest to Normal (Highest Cos Theta)
            best_tgt = None
            best_cos = -1.0

            for tgt in visible_targets:
                # Re-calculate cos theta locally
                # This is duplicated logic but fast
                az_rad = np.radians(tgt[0])
                el_rad = np.radians(tgt[1])
                u = np.cos(el_rad) * np.cos(az_rad)
                v = np.cos(el_rad) * np.sin(az_rad)
                w = np.sin(el_rad)
                g_dir = np.array([u, v, w])
                l_dir = panel.rotation_matrix.T @ g_dir
                cos_theta = l_dir[2]

                if cos_theta > best_cos:
                    best_cos = cos_theta
                    best_tgt = tgt

            # Select Nulls (Others in FoV)
            nulls = [t for t in visible_targets if t != best_tgt]

            # Configure RF Chain
            chain.set_beam(best_tgt[0], best_tgt[1], nulls)

        # 2. Baseband Processing (Stream Creation)
        # Create one stream per hypothesized target
        results = {}

        for i, target in enumerate(hypothesized_targets):
            # Define Interferers: All other targets
            interferers = [t for j, t in enumerate(hypothesized_targets) if j != i]

            # Optimize Baseband Weights for this stream
            # Note: This creates a NEW set of weights for this stream.
            # The 'self.system.baseband' object is shared, so we overwrite weights sequentially.
            # In a real system, we'd have N parallel beamformers.

            # We pass explicit interferers.
            # Should we also pass "RF Risk Angles"?
            # The prompt says: "inference part... check measurement results with its risk angles".
            # But here "distribution part... baseband should create new high purity beams".
            # Let's stick to explicit targets as interferers for now.

            weights = self.system.baseband.optimize_weights(self.system.rf_chains, target, interferers)

            # Analyze Stream
            metrics = self.system.analyze_system_performance(target[0], target[1], interferers)

            # Store weights in results for evaluation
            # Convert to list for JSON serialization if needed later, but here numpy is fine
            metrics['weights'] = weights

            results[i] = metrics

        return results
