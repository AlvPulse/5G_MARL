import numpy as np
from typing import Tuple, Optional, List

class SparsePlanarPanel:
    """
    Represents a Sparse Uniform Planar Array (UPA) positioned and oriented in 3D space.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        frequency: float,
        sparsity_mask: Optional[np.ndarray] = None,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float] = (0.0, 90.0)  # (az, el) in degrees
    ):
        """
        Initialize the panel.

        Args:
            rows: Number of rows in the full grid.
            cols: Number of columns in the full grid.
            frequency: Operating frequency in Hz.
            sparsity_mask: Boolean matrix (rows, cols). True = element present.
                           If None, assumes full array.
            position: Global (x, y, z) position of the center of the array.
            orientation: Boresight direction (azimuth, elevation) in degrees.
                         Azimuth: 0 is +X, 90 is +Y.
                         Elevation: 0 is XY plane, 90 is +Z.
        """
        self.rows = rows
        self.cols = cols
        self.frequency = frequency
        self.c = 299792458.0  # Speed of light
        self.wavelength = self.c / frequency
        self.d = 0.5 * self.wavelength  # Half-wavelength spacing
        self.position = np.array(position)

        # Parse orientation
        self.az_deg, self.el_deg = orientation

        # Handle sparsity mask
        if sparsity_mask is None:
            self.mask = np.ones((rows, cols), dtype=bool)
        else:
            self.mask = np.array(sparsity_mask, dtype=bool)
            if self.mask.shape != (rows, cols):
                raise ValueError(f"Sparsity mask shape {self.mask.shape} does not match grid dimensions ({rows}, {cols}).")

        # 1. Generate Local Grid (centered at 0,0,0 in xy-plane)
        # We center the grid so rotation is around the centroid
        y_idx = np.arange(rows) - (rows - 1) / 2.0
        x_idx = np.arange(cols) - (cols - 1) / 2.0

        # Meshgrid: x corresponds to cols, y to rows
        # Note: In array processing, often x is dim 0, y is dim 1 or vice versa.
        # Let's map cols to local x-axis and rows to local y-axis.
        xx, yy = np.meshgrid(x_idx, y_idx)

        # Apply spacing
        xx = xx * self.d
        yy = yy * self.d
        zz = np.zeros_like(xx)

        # Flatten and apply mask
        # We only store positions of active elements
        self.local_positions = np.stack([xx[self.mask], yy[self.mask], zz[self.mask]], axis=1) # (N_active, 3)
        self.num_elements = self.local_positions.shape[0]

        # 2. Compute Rotation Matrix
        self.rotation_matrix = self._compute_rotation_matrix()

        # 3. Rotate and Translate to Global Coordinates
        self.global_positions = self._transform_to_global(self.local_positions)

    def _compute_rotation_matrix(self) -> np.ndarray:
        """
        Computes the rotation matrix R to transform local to global frame.
        """
        az_rad = np.radians(self.az_deg)
        pitch_rad = np.radians(90.0 - self.el_deg)

        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0,                 1, 0],
            [-np.sin(pitch_rad),0, np.cos(pitch_rad)]
        ])

        Rz = np.array([
            [np.cos(az_rad), -np.sin(az_rad), 0],
            [np.sin(az_rad),  np.cos(az_rad), 0],
            [0,               0,              1]
        ])

        return Rz @ Ry

    def _transform_to_global(self, local_pos: np.ndarray) -> np.ndarray:
        """
        Rotates and translates local positions to global frame.
        """
        # Apply rotation: pos_global = R @ pos_local.T
        rotated_pos = (self.rotation_matrix @ local_pos.T).T

        # Apply translation
        global_pos = rotated_pos + self.position

        return global_pos

    def get_steering_vector(self, az_deg: float, el_deg: float) -> np.ndarray:
        """
        Calculates the steering vector for a given GLOBAL direction.
        Includes Element Pattern.
        """
        az_rad = np.radians(az_deg)
        el_rad = np.radians(el_deg)

        u = np.cos(el_rad) * np.cos(az_rad)
        v = np.cos(el_rad) * np.sin(az_rad)
        w = np.sin(el_rad)

        direction_vector = np.array([u, v, w])

        k_vec = (2 * np.pi / self.wavelength) * direction_vector

        # Phase delays
        phases = -np.dot(self.global_positions, k_vec)
        sv = np.exp(1j * phases)

        # Apply element pattern
        pattern = self.get_element_pattern(az_deg, el_deg)
        sv = sv * pattern

        return sv

    def get_element_pattern(self, az_deg: float, el_deg: float) -> float:
        """
        Returns the element gain (magnitude) for a given direction.
        Uses Cosine pattern based on angle from Panel Normal.
        """
        # Global direction vector
        az_rad = np.radians(az_deg)
        el_rad = np.radians(el_deg)

        u = np.cos(el_rad) * np.cos(az_rad)
        v = np.cos(el_rad) * np.sin(az_rad)
        w = np.sin(el_rad)

        global_dir = np.array([u, v, w])

        # Transform to local frame: v_local = R^T * v_global
        local_dir = self.rotation_matrix.T @ global_dir

        # Local normal is Z (0,0,1). Cos theta = local_dir[2]
        cos_theta = local_dir[2]

        # Back lobe suppression: if cos_theta < 0, return small gain or 0
        if cos_theta > 0:
            return float(cos_theta) ** 1.0 # Cosine pattern power 1
        else:
            return 0.0 # Strict front-only
