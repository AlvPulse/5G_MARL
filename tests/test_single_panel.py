import numpy as np
import pytest
from src.geometry import SparsePlanarPanel
from src.algorithms import IterativePOCSSolver
from src.rf_chain import RFChain

def test_single_panel_physics():
    """
    Verification script for Single Panel Physics.
    - 8x8 Sparse Panel (20% random sparsity).
    - Target (0, 10), Nulls [(-20, 10), (20, 10)].
    - IterativePOCSSolver.
    - Assert Min Null Depth > 30dB.
    """

    # 1. Setup Panel
    rows, cols = 8, 8
    freq = 28e9 # 28 GHz

    # create deterministic sparsity mask (20% sparse)
    np.random.seed(42)
    total_elements = rows * cols
    num_inactive = int(0.2 * total_elements)
    mask_flat = np.ones(total_elements, dtype=bool)
    inactive_indices = np.random.choice(total_elements, num_inactive, replace=False)
    mask_flat[inactive_indices] = False
    mask = mask_flat.reshape((rows, cols))

    panel = SparsePlanarPanel(
        rows=rows,
        cols=cols,
        frequency=freq,
        sparsity_mask=mask,
        position=(0,0,0),
        orientation=(0, 90) # Facing +Z (Standard)
    )

    # 2. Setup Solver and RF Chain
    solver = IterativePOCSSolver(max_iterations=100)
    rf_chain = RFChain(panel, solver)

    # 3. Define Scenario
    target = (0, 10) # Az=0, El=10
    nulls = [(-20, 10), (20, 10)]

    # 4. Set Beam
    print(f"Steering to {target}, Nulling {nulls}")
    rf_chain.set_beam(target[0], target[1], nulls)

    # 5. Analyze
    metrics = rf_chain.analyze_performance(target[0], target[1], nulls, scan_res=1.0)

    print("\nPerformance Metrics:")
    print(f"Target Gain: {metrics['target_gain_db']:.2f} dB")
    print(f"Min Null Depth: {metrics['min_null_depth_db']:.2f} dB")
    print(f"Peak SLL: {metrics['peak_sll_db']:.2f} dB")
    print(f"Risk Angles: {metrics['risk_angles']}")

    # 6. Assertions
    # Target gain should be reasonably high (approx 10*log10(N_active))
    # N_active = 64 - 12 = 52. Gain ~ 17 dB + element gain?
    # Array Factor magnitude is sum(1) = N. Power is N^2.
    # 10log(52^2) = 20log(52) = 34 dB.
    assert metrics['target_gain_db'] > 25.0, "Target gain unexpectedly low."

    # Null Depth Check
    # Requirement: > 30dB
    assert metrics['min_null_depth_db'] > 30.0, f"Null depth {metrics['min_null_depth_db']:.2f} dB is insufficient (Wanted > 30dB)."

if __name__ == "__main__":
    try:
        test_single_panel_physics()
        print("\nTEST PASSED")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
