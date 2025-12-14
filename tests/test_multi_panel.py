import numpy as np
import pytest
from src.geometry import SparsePlanarPanel
from src.algorithms import IterativePOCSSolver
from src.rf_chain import RFChain
from src.baseband import BasebandProcessor
from src.system import MultiPanelSystem

def test_multi_panel_optimization():
    """
    Verify that Baseband Optimization improves performance.
    Setup:
    - 4 Panels (Facing N, S, E, W).
    - Target at (0, 10).
    - Null at (20, 10).
    - Interferer at (45, 10) (Simulating a Risk Angle).
    """

    # 1. Setup Panels (4 Sides of a building?)
    # North (Az=0), East (Az=90), South (Az=180), West (Az=270)
    # Positions shifted slightly from origin
    panels = []
    chains = []

    orientations = [(0, 90), (90, 90), (180, 90), (270, 90)]
    positions = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)] # 1m spacing

    rows, cols = 4, 4 # Small panels for speed
    freq = 28e9

    for i in range(4):
        p = SparsePlanarPanel(
            rows=rows, cols=cols, frequency=freq,
            position=positions[i],
            orientation=orientations[i]
        )
        s = IterativePOCSSolver(max_iterations=20)
        c = RFChain(p, s)
        panels.append(p)
        chains.append(c)

    baseband = BasebandProcessor()
    system = MultiPanelSystem(panels, chains, baseband)

    target = (0, 10)
    nulls = [(20, 10)]
    interferers = [(20, 10), (45, 10)] # Null + Another source

    # 2. Configure RF Chains (Sub-optimal or Standard)
    # Let's say we steer all of them to target.
    # Some panels (like South) might not see the target well, but they try.
    print("Steering RF Chains...")
    for c in chains:
        c.set_beam(target[0], target[1], nulls)

    # Check single panel performance (e.g. Panel 0)
    metrics0 = chains[0].analyze_performance(target[0], target[1], nulls)
    print(f"Panel 0 Null Depth: {metrics0['min_null_depth_db']:.2f} dB")

    # 3. Run Baseband Optimization
    print("Optimizing Baseband...")
    # We ask baseband to null the explicit nulls AND the extra interferer
    baseband.optimize_weights(chains, target, interferers)

    print("Baseband Weights:", baseband.weights)

    # 4. Evaluate System
    sys_metrics = system.analyze_system_performance(target[0], target[1], interferers)

    print("\nSystem Performance:")
    print(f"Target Gain: {sys_metrics['target_gain_db']:.2f} dB")
    print(f"Min Null Depth: {sys_metrics['min_null_depth_db']:.2f} dB")

    # 5. Assert Improvement
    # System null depth should be better than single panel (or at least very good)
    # The optimization should force null depth to be very high (mathematically infinite if DOFs allow)
    # With 4 chains, we can handle 1 target + 2 nulls easily.

    assert sys_metrics['min_null_depth_db'] > 40.0, "Baseband failed to achieve high null depth."
    assert sys_metrics['target_gain_db'] > 10.0, "Baseband killed the target gain."

if __name__ == "__main__":
    try:
        test_multi_panel_optimization()
        print("\nTEST PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
