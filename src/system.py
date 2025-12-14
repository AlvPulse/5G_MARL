import numpy as np
from typing import List, Tuple, Dict
from scipy.ndimage import maximum_filter

from src.geometry import SparsePlanarPanel
from src.rf_chain import RFChain
from src.baseband import BasebandProcessor

class MultiPanelSystem:
    """
    Manages multiple RF chains and a Baseband Processor.
    """

    def __init__(self, panels: List[SparsePlanarPanel], rf_chains: List[RFChain], baseband: BasebandProcessor):
        self.panels = panels
        self.rf_chains = rf_chains
        self.baseband = baseband

    def analyze_system_performance(
        self,
        target_az: float,
        target_el: float,
        nulls_list: List[Tuple[float, float]],
        scan_res: float = 1.0,
        el_limit: float = 90.0
    ) -> Dict:
        """
        Scans the composite system beam pattern.
        """
        weights = self.baseband.weights
        if weights is None:
             raise ValueError("Baseband weights not set.")

        # 1. Create Scan Grid (Same as RFChain)
        az_scan = np.arange(0, 360, scan_res)
        el_scan = np.arange(0, el_limit + scan_res, scan_res)
        az_grid, el_grid = np.meshgrid(az_scan, el_scan)
        az_flat = az_grid.flatten()
        el_flat = el_grid.flatten()

        # 2. Compute Composite Response
        # R_total(theta) = Sum_i ( w_bb_i * AF_i(theta) )

        # We need AF_i(theta) for all i, for all theta.
        # Let's reuse vectorization logic from RFChain but for multiple panels.
        # This is computationally heavy.

        # Pre-compute direction vectors (u,v,w)
        u = np.cos(np.radians(el_flat)) * np.cos(np.radians(az_flat))
        v = np.cos(np.radians(el_flat)) * np.sin(np.radians(az_flat))
        w = np.sin(np.radians(el_flat))

        # Shape (N_points, 3)
        dir_vecs = np.stack([u, v, w], axis=1)

        total_response = np.zeros(az_flat.shape, dtype=complex)

        for i, chain in enumerate(self.rf_chains):
            # Calculate AF_i for all points
            k_vecs = (2 * np.pi / chain.panel.wavelength) * dir_vecs
            phases = -np.dot(chain.panel.global_positions, k_vecs.T) # (N_elem, N_points)
            A_scan = np.exp(1j * phases)

            # AF_i: (1, N_points)
            af_i = chain.current_weights.conj() @ A_scan

            # Add to total with baseband weight
            total_response += weights[i] * af_i

        # 3. Metrics (Copied logic from RFChain)
        gain_linear = np.abs(total_response)**2
        gain_db = 10 * np.log10(gain_linear + 1e-12)
        gain_db_grid = gain_db.reshape(az_grid.shape)

        # Target Gain
        # We can just pick the value at the closest grid point or recompute exact
        # Recompute exact for accuracy
        exact_response = 0j
        for i, chain in enumerate(self.rf_chains):
             exact_response += weights[i] * chain.get_complex_response(target_az, target_el)
        target_gain_db = 10 * np.log10(np.abs(exact_response)**2 + 1e-12)

        # Null Depth
        null_gains = []
        for naz, nel in nulls_list:
            n_resp = 0j
            for i, chain in enumerate(self.rf_chains):
                n_resp += weights[i] * chain.get_complex_response(naz, nel)
            n_gain_db = 10 * np.log10(np.abs(n_resp)**2 + 1e-12)
            null_gains.append(n_gain_db)

        if null_gains:
            max_null_gain = max(null_gains)
            min_null_depth_db = target_gain_db - max_null_gain
        else:
            min_null_depth_db = float('inf')

        # Peak SLL
        d_az = np.radians(az_grid - target_az)
        t_el_rad = np.radians(target_el)
        g_el_rad = np.radians(el_grid)
        arg = np.sin(t_el_rad) * np.sin(g_el_rad) + np.cos(t_el_rad) * np.cos(g_el_rad) * np.cos(d_az)
        arg = np.clip(arg, -1.0, 1.0)
        dist_deg = np.degrees(np.arccos(arg))

        main_lobe_radius = 15.0
        sll_mask = dist_deg > main_lobe_radius
        sll_region_gains = gain_db_grid[sll_mask]

        if sll_region_gains.size > 0:
            peak_sll_val = np.max(sll_region_gains)
            peak_sll_db = target_gain_db - peak_sll_val
        else:
            peak_sll_db = 0.0

        return {
            "target_gain_db": float(target_gain_db),
            "min_null_depth_db": float(min_null_depth_db),
            "peak_sll_db": float(peak_sll_db)
        }
