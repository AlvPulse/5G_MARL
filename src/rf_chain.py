import numpy as np
from typing import List, Tuple, Dict
from scipy.ndimage import maximum_filter

from src.geometry import SparsePlanarPanel
from src.algorithms import BeamformerSolver

class RFChain:
    """
    Represents an RF Chain consisting of a Panel and a Solver.
    """

    def __init__(self, panel: SparsePlanarPanel, solver: BeamformerSolver):
        self.panel = panel
        self.solver = solver
        # Initialize weights to all 1s (0 phase)
        self.current_weights = np.ones(self.panel.num_elements, dtype=complex)

    def set_beam(self, target_az: float, target_el: float, nulls_list: List[Tuple[float, float]]):
        """
        Calculates and updates weights to steer beam to target and null specified directions.

        Args:
            target_az, target_el: Coordinates of the target.
            nulls_list: List of (az, el) tuples for interference nulling.
        """
        # 1. Collect Constraints
        # Target: Gain = N (Max possible for Phase-Only)
        # We scale the target constraint to match the natural array gain
        # to avoid the solver fighting the phase constraint.
        constraints_dirs = [(target_az, target_el)]
        constraints_vals = [float(self.panel.num_elements) + 0j]

        # Nulls: Gain = 0
        for null_az, null_el in nulls_list:
            constraints_dirs.append((null_az, null_el))
            constraints_vals.append(0.0 + 0j)

        # 2. Build Matrix A (Steering Vectors) and Vector d
        # A should be (N_elements, N_constraints)
        steering_vectors = []
        for az, el in constraints_dirs:
            sv = self.panel.get_steering_vector(az, el)
            steering_vectors.append(sv)

        A = np.stack(steering_vectors, axis=1) # Stack as columns
        d = np.array(constraints_vals, dtype=complex)

        # 3. Solve
        self.current_weights = self.solver.solve(A, d, self.current_weights)

    def analyze_performance(
        self,
        target_az: float,
        target_el: float,
        nulls_list: List[Tuple[float, float]],
        scan_res: float = 1.0,
        el_limit: float = 90.0
    ) -> Dict:
        """
        Scans the beam pattern to compute performance metrics.

        Args:
            target_az, target_el: Expected target location.
            nulls_list: Expected null locations.
            scan_res: Resolution of the scan grid in degrees.
            el_limit: Maximum elevation angle magnitude to scan (0 to 90).

        Returns:
            Dictionary with metrics: target_gain_db, min_null_depth_db, peak_sll_db, risk_angles.
        """
        # 1. Create Scan Grid
        # Azimuth: 0 to 360
        az_scan = np.arange(0, 360, scan_res)
        # Elevation: 0 to el_limit (since we defined el from XY plane up)
        # Assuming we scan the upper hemisphere or limited range
        # Note: If panel is tilted, we might want to scan relative to panel or global.
        # Requirement implies global scan.
        el_scan = np.arange(0, el_limit + scan_res, scan_res)

        # We need 2D grid
        az_grid, el_grid = np.meshgrid(az_scan, el_scan)

        # Flatten for vectorized computation
        az_flat = az_grid.flatten()
        el_flat = el_grid.flatten()

        # 2. Compute Array Factor Response
        # We can optimize this by batching.
        # response = w^H * a(theta, phi)
        # To do this efficiently:
        # Construct large matrix of steering vectors A_scan (N_elem, N_points)
        # response = w^H @ A_scan

        # However, calling get_steering_vector for every point might be slow if not optimized.
        # Optimization:
        # SV calculation: exp(j * k dot p)
        # k depends on (az, el). p is constant.
        # p is (N_elem, 3)
        # k_matrix is (N_points, 3)
        # phases = - (p @ k_matrix.T) -> (N_elem, N_points)

        # Let's vectorize k calculation
        u = np.cos(np.radians(el_flat)) * np.cos(np.radians(az_flat))
        v = np.cos(np.radians(el_flat)) * np.sin(np.radians(az_flat))
        w = np.sin(np.radians(el_flat))

        k_vecs = (2 * np.pi / self.panel.wavelength) * np.stack([u, v, w], axis=1) # (N_points, 3)

        # phases: (N_elem, N_points)
        # self.panel.global_positions: (N_elem, 3)
        phases = -np.dot(self.panel.global_positions, k_vecs.T)

        # Steering Matrix A_scan
        A_scan = np.exp(1j * phases)

        # Beam Pattern (Voltage)
        # response: (1, N_points)
        response = self.current_weights.conj() @ A_scan

        # Power Gain
        gain_linear = np.abs(response)**2
        # Normalize to element count (optional, but usually Array Factor is max N^2)
        # Let's stick to raw gain or normalize to max?
        # Standard: 10*log10(|AF|^2).
        # If weights are normalized to 1, max gain is N^2.
        gain_db = 10 * np.log10(gain_linear + 1e-12)

        # Reshape to grid
        gain_db_grid = gain_db.reshape(az_grid.shape)

        # 3. Extract Metrics

        # A. Target Gain
        # Find closest point in grid to target
        # Or just compute exactly at target (more accurate)
        target_sv = self.panel.get_steering_vector(target_az, target_el)
        target_resp = np.dot(self.current_weights.conj(), target_sv)
        target_gain_db = 10 * np.log10(np.abs(target_resp)**2 + 1e-12)

        # B. Null Depth
        # Calculate gain at exact null locations
        null_gains = []
        for naz, nel in nulls_list:
            n_sv = self.panel.get_steering_vector(naz, nel)
            n_resp = np.dot(self.current_weights.conj(), n_sv)
            n_gain_db = 10 * np.log10(np.abs(n_resp)**2 + 1e-12)
            null_gains.append(n_gain_db)

        if null_gains:
            max_null_gain = max(null_gains)
            # Depth is relative to target gain
            min_null_depth_db = target_gain_db - max_null_gain
        else:
            min_null_depth_db = float('inf')

        # C. Peak SLL & Risk Angles
        # We need to mask the main lobe.
        # Simple heuristic: Find peak location (should be near target).
        # Mask out a region around target (e.g., +/- beamwidth).
        # Approx beamwidth for UPA is ~ 100 / sqrt(N) degrees or similar.
        # Let's assume a fixed angular radius for main lobe exclusion.
        # Radius = 10 degrees (Generous for user discovery arrays).

        # Distance on sphere approximation (great circle)
        # dist = arccos( sin(el1)sin(el2) + cos(el1)cos(el2)cos(az1-az2) )

        d_az = np.radians(az_grid - target_az)
        t_el_rad = np.radians(target_el)
        g_el_rad = np.radians(el_grid)

        # Central angle formula
        # Clamp arg to [-1, 1] to avoid numerical errors
        arg = np.sin(t_el_rad) * np.sin(g_el_rad) + np.cos(t_el_rad) * np.cos(g_el_rad) * np.cos(d_az)
        arg = np.clip(arg, -1.0, 1.0)
        dist_deg = np.degrees(np.arccos(arg))

        # Main lobe mask (exclude points within X degrees)
        # For a standard array, BW ~ 50 deg / (L/lambda). 8x8 -> L=4lam -> 12 deg.
        # Let's exclude 15 degrees.
        main_lobe_radius = 15.0
        sll_mask = dist_deg > main_lobe_radius

        sll_region_gains = gain_db_grid[sll_mask]

        if sll_region_gains.size > 0:
            peak_sll_val = np.max(sll_region_gains)
            peak_sll_db = target_gain_db - peak_sll_val
        else:
            peak_sll_db = 0.0 # Should not happen unless huge radius

        # Find Risk Angles (peaks in SLL region that are high)
        # "Risk" means SLL is bad (low difference).
        # Let's define risk as any lobe within 3dB of the highest SLL peak.
        # Using maximum_filter to find local maxima

        local_max = maximum_filter(gain_db_grid, size=5) == gain_db_grid
        # Combine with SLL mask
        risk_mask = local_max & sll_mask

        # Get indices
        risk_indices = np.argwhere(risk_mask)
        risk_angles = []

        # Sort by gain descending
        risk_gains = gain_db_grid[risk_mask]
        sorted_idxs = np.argsort(risk_gains)[::-1]

        # Return top 5 risk angles
        for i in sorted_idxs[:5]:
            r_idx = risk_indices[i]
            r_az = az_grid[r_idx[0], r_idx[1]]
            r_el = el_grid[r_idx[0], r_idx[1]]
            risk_angles.append((float(r_az), float(r_el)))

        return {
            "target_gain_db": float(target_gain_db),
            "min_null_depth_db": float(min_null_depth_db),
            "peak_sll_db": float(peak_sll_db),
            "risk_angles": risk_angles
        }
