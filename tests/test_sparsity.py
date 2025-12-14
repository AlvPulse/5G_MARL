import numpy as np
import pytest
from src.geometry import SparsePlanarPanel
from src.algorithms import IterativePOCSSolver
from src.rf_chain import RFChain
from src.baseband import BasebandProcessor
from src.system import MultiPanelSystem

def test_sparsity_scenarios():
    """
    Test suite for High Sparsity Scenarios.
    Panel Config: 16x4, 65% Sparsity.
    """
    rows, cols = 16, 4
    sparsity_ratio = 0.65
    freq = 28e9

    # Deterministic Sparsity Mask
    np.random.seed(42)
    total_elements = rows * cols
    num_inactive = int(sparsity_ratio * total_elements)
    mask_flat = np.ones(total_elements, dtype=bool)
    inactive_indices = np.random.choice(total_elements, num_inactive, replace=False)
    mask_flat[inactive_indices] = False
    mask = mask_flat.reshape((rows, cols))

    # ------------------------------------------------------------------------
    # Scenario A: Single Panel Mode
    # ------------------------------------------------------------------------
    print("\n--- Scenario A: Single Panel (16x4, 65% Sparse) ---")
    panel = SparsePlanarPanel(
        rows, cols, freq, mask,
        position=(0,0,0), orientation=(0, 90) # Face +Z
    )
    solver = IterativePOCSSolver(max_iterations=50)
    rf_chain = RFChain(panel, solver)

    target = (0, 10) # 80 deg off boresight
    nulls = [(-20, 10)]

    rf_chain.set_beam(target[0], target[1], nulls)
    metrics_a = rf_chain.analyze_performance(target[0], target[1], nulls)

    print(f"Target Gain: {metrics_a['target_gain_db']:.2f} dB")
    print(f"Null Depth: {metrics_a['min_null_depth_db']:.2f} dB")

    # Assertions for Scenario A
    # With 65% sparsity, 64 elements -> ~22 active.
    # Gain ~ 10log(22^2) = 26 dB.
    # Element pattern loss at 80 deg off boresight: cos(80) = 0.17 (-15 dB).
    # Net gain expected: 26 - 15 = 11 dB.
    assert metrics_a['min_null_depth_db'] > 20.0, "Scenario A: Null depth failed."

    # ------------------------------------------------------------------------
    # Scenario B: Multi Panel Single Target
    # ------------------------------------------------------------------------
    print("\n--- Scenario B: Multi Panel (4 Panels) Single Target ---")
    # Setup 4 panels
    orientations = [(0, 90), (90, 90), (180, 90), (270, 90)]
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

    # Target (0, 10)
    # Panel 0 (Face N/X) sees it best? No, Panel 0 is (0, 90) facing +Z?
    # Wait, in previous tests I assumed (0, 90) is facing +Z.
    # BUT, if I want panels facing N, E, S, W horizontally:
    # N: Az=0, El=0.
    # User said "elevation... limited up to 30 degree".
    # Usually panels are vertical.
    # Let's assume panels are mounted vertically, facing horizon.
    # Orientation: (0, 0), (90, 0), (180, 0), (270, 0).
    # Re-setup for realistic building scenario

    panels_horiz = []
    chains_horiz = []
    orientations_horiz = [(0, 0), (90, 0), (180, 0), (270, 0)]

    for i in range(4):
        p = SparsePlanarPanel(rows, cols, freq, mask, positions[i], orientations_horiz[i])
        s = IterativePOCSSolver(max_iterations=30)
        c = RFChain(p, s)
        panels_horiz.append(p)
        chains_horiz.append(c)

    system_b = MultiPanelSystem(panels_horiz, chains_horiz, baseband)

    target_b = (0, 10) # Az=0, El=10. Panel 0 should see it well. Panel 2 (180) is back.
    nulls_b = [(30, 10)]

    # Configure RF
    for c in chains_horiz:
        c.set_beam(target_b[0], target_b[1], nulls_b)

    # Optimize Baseband
    baseband.optimize_weights(chains_horiz, target_b, nulls_b)

    metrics_b = system_b.analyze_system_performance(target_b[0], target_b[1], nulls_b)

    print(f"System Gain: {metrics_b['target_gain_db']:.2f} dB")
    print(f"System Null Depth: {metrics_b['min_null_depth_db']:.2f} dB")

    assert metrics_b['min_null_depth_db'] > 40.0, "Scenario B: System null depth failed."

    # ------------------------------------------------------------------------
    # Scenario C: Multi Panel Multi User (MU-MIMO)
    # ------------------------------------------------------------------------
    print("\n--- Scenario C: Multi Panel Multi User ---")
    # User 1: (0, 10) -> Served by Panel 0 (0 deg)
    # User 2: (90, 10) -> Served by Panel 1 (90 deg)

    u1 = (0.0, 10.0)
    u2 = (90.0, 10.0)

    # Strategy:
    # Panels 0 & 3 try to serve U1? Or just Panel 0?
    # Panel 0 -> Target U1, Null U2.
    # Panel 1 -> Target U2, Null U1.
    # Panel 2 (180) -> Sees neither well. Target U1?
    # Panel 3 (270) -> Sees U1 better?

    # Let's assign:
    # Panel 0: Target U1, Null U2
    # Panel 1: Target U2, Null U1
    # Panel 2: Target U1, Null U2 (Backside, negligible)
    # Panel 3: Target U1, Null U2 (Side)

    print("Configuring RF Chains for assigned users...")
    chains_horiz[0].set_beam(u1[0], u1[1], [u2])
    chains_horiz[1].set_beam(u2[0], u2[1], [u1])
    chains_horiz[2].set_beam(u1[0], u1[1], [u2])
    chains_horiz[3].set_beam(u1[0], u1[1], [u2])

    # Baseband Optimization 1: Stream for U1
    # We want High Gain at U1, Null at U2.
    bb_u1 = BasebandProcessor()
    w_u1 = bb_u1.optimize_weights(chains_horiz, u1, [u2])

    # Baseband Optimization 2: Stream for U2
    # We want High Gain at U2, Null at U1.
    bb_u2 = BasebandProcessor()
    w_u2 = bb_u2.optimize_weights(chains_horiz, u2, [u1])

    # Evaluate Stream 1
    system_b.baseband.weights = w_u1 # Inject weights
    metrics_u1 = system_b.analyze_system_performance(u1[0], u1[1], [u2])
    print(f"Stream 1 (Target U1): Gain {metrics_u1['target_gain_db']:.2f} dB, Null U2 {metrics_u1['min_null_depth_db']:.2f} dB")

    # Evaluate Stream 2
    system_b.baseband.weights = w_u2 # Inject weights
    metrics_u2 = system_b.analyze_system_performance(u2[0], u2[1], [u1])
    print(f"Stream 2 (Target U2): Gain {metrics_u2['target_gain_db']:.2f} dB, Null U1 {metrics_u2['min_null_depth_db']:.2f} dB")

    # Assertions
    # Stream 1 should have high gain at U1
    assert metrics_u1['target_gain_db'] > 10.0, "Scenario C: Stream 1 Gain Low"
    assert metrics_u1['min_null_depth_db'] > 30.0, "Scenario C: Stream 1 Interference High"

    # Stream 2 should have high gain at U2
    assert metrics_u2['target_gain_db'] > 10.0, "Scenario C: Stream 2 Gain Low"
    assert metrics_u2['min_null_depth_db'] > 30.0, "Scenario C: Stream 2 Interference High"


if __name__ == "__main__":
    try:
        test_sparsity_scenarios()
        print("\nTESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
