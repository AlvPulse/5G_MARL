import numpy as np
import pytest
from src.geometry import SparsePlanarPanel
from src.algorithms import IterativePOCSSolver
from src.rf_chain import RFChain
from src.baseband import BasebandProcessor
from src.system import MultiPanelSystem
from src.brain import BeamformingBrain

def test_brain_distribution():
    """
    Test the Brain's ability to distribute targets and create streams.
    Setup:
    - 4 Panels (Horizontal): N (0), E (90), S (180), W (270).
    - 3 Targets:
      - T1: (0, 10)  -> Seen by N (Primary), E, W (Side).
      - T2: (90, 10) -> Seen by E (Primary), N, S.
      - T3: (180, 10)-> Seen by S (Primary), E, W.
    """

    # 1. Setup System
    rows, cols = 16, 4
    freq = 28e9

    # Deterministic Sparsity (50%)
    np.random.seed(42)
    total = rows * cols
    mask_flat = np.ones(total, dtype=bool)
    mask_flat[np.random.choice(total, int(0.5*total), replace=False)] = False
    mask = mask_flat.reshape((rows, cols))

    orientations = [(0, 0), (90, 0), (180, 0), (270, 0)]
    positions = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]

    panels = []
    chains = []
    for i in range(4):
        p = SparsePlanarPanel(rows, cols, freq, mask, positions[i], orientations[i])
        s = IterativePOCSSolver(max_iterations=30)
        c = RFChain(p, s)
        panels.append(p)
        chains.append(c)

    baseband = BasebandProcessor()
    system = MultiPanelSystem(panels, chains, baseband)
    brain = BeamformingBrain(system)

    # 2. Define Targets
    hypothesized_targets = [
        (0.0, 10.0),   # T0
        (90.0, 10.0),  # T1
        (180.0, 10.0)  # T2
    ]

    print("\n--- Running Brain Distribution ---")
    results = brain.distribute_and_process(hypothesized_targets, fov_deg=60.0)

    # 3. Analyze Results
    for i, res in results.items():
        print(f"\nStream {i} (Target {hypothesized_targets[i]}):")
        print(f"  Gain: {res['target_gain_db']:.2f} dB")
        print(f"  Null Depth: {res['min_null_depth_db']:.2f} dB")

        # Verify Performance
        assert res['target_gain_db'] > 5.0, f"Stream {i}: Low Gain"
        assert res['min_null_depth_db'] > 25.0, f"Stream {i}: Poor Nulling"

    # Verify Panel Assignment Logic (Implicitly)
    # Check RF Chains directly to see what they did
    # Panel 0 (Az 0) should have targeted T0 (0, 10) and nulled T1/T2?
    # T1 (90) is 90 deg away -> Not in FoV (60 deg).
    # T2 (180) is 180 deg away -> Not in FoV.
    # So Panel 0 should only see T0.

    # Panel 1 (Az 90) sees T1 (Primary), T0 (Side 90 deg? No), T2 (Side 90 deg? No).
    # Wait, (0, 10) relative to (90, 0).
    # Az diff 90. Cos theta ~ cos(90) = 0.
    # So T0 is on the edge of visibility for Panel 1.
    # With fov=60, Panel 1 only sees T1.

    # So each panel should have picked its respective target.
    # Panel 3 (270) sees nothing? (0 is 90 away, 90 is 180 away, 180 is 90 away).
    # Panel 3 should be idle (Zenith).

    # Let's check weights/gain of Panel 3 at Zenith
    p3_zenith = chains[3].analyze_performance(0, 90, [])
    # Should have high gain at 0, 90
    print(f"\nPanel 3 (Idle) Zenith Gain: {p3_zenith['target_gain_db']:.2f} dB")

if __name__ == "__main__":
    try:
        test_brain_distribution()
        print("\nTEST PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
