import numpy as np
import json
import os
from typing import List, Dict

from src.geometry import SparsePlanarPanel
from src.algorithms import IterativePOCSSolver
from src.rf_chain import RFChain
from src.baseband import BasebandProcessor
from src.system import MultiPanelSystem

def generate_dataset(
    num_samples: int = 10,
    output_file: str = "dataset.jsonl"
):
    """
    Generates a dataset of beamforming scenarios.
    """

    # 1. System Setup (Static)
    # 4 Panels (N, E, S, W)
    # Use horizontal mounting (El=0) for realistic building coverage
    orientations = [(0, 0), (90, 0), (180, 0), (270, 0)]
    positions = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]

    # User Requirement: Rectangular 16x4 Arrays
    rows, cols = 16, 4
    freq = 28e9

    # Create Panels
    panels = []
    for i in range(4):
        # 50% Sparsity (More challenging)
        total = rows * cols
        mask_flat = np.ones(total, dtype=bool)
        mask_flat[np.random.choice(total, int(0.5*total), replace=False)] = False
        mask = mask_flat.reshape((rows, cols))

        p = SparsePlanarPanel(rows, cols, freq, mask, positions[i], orientations[i])
        panels.append(p)

    print(f"Generating {num_samples} samples...")

    with open(output_file, 'w') as f:
        for sample_idx in range(num_samples):
            # 2. Random Scenario
            # Target in reasonable FOV (e.g., Az -60 to 60 relative to global? No, global 360)
            # Let's pick random Az 0-360, El 5-45
            t_az = np.random.uniform(0, 360)
            t_el = np.random.uniform(5, 45)
            target = (t_az, t_el)

            # Interferers (1 to 3)
            num_int = np.random.randint(1, 4)
            interferers = []
            for _ in range(num_int):
                # Ensure not too close to target (min 10 deg separation)
                while True:
                    i_az = np.random.uniform(0, 360)
                    i_el = np.random.uniform(5, 45)
                    # Check dist
                    if abs(i_az - t_az) > 10 or abs(i_el - t_el) > 10:
                        interferers.append((i_az, i_el))
                        break

            # 3. RF Stage
            chains = []
            rf_metrics_list = []
            risk_angles_all = []

            # Re-create solvers/chains fresh or reset them
            for p in panels:
                # Use fewer iterations for speed in dataset gen
                solver = IterativePOCSSolver(max_iterations=30)
                chain = RFChain(p, solver)

                # RF Agent Action: Point to Target, Null known Interferers
                chain.set_beam(t_az, t_el, interferers)

                # RF Analysis
                m = chain.analyze_performance(t_az, t_el, interferers)
                rf_metrics_list.append(m)
                risk_angles_all.extend(m['risk_angles'])
                chains.append(chain)

            # 4. Baseband Stage
            baseband = BasebandProcessor()
            system = MultiPanelSystem(panels, chains, baseband)

            # Optimization Inputs: Target, Explicit Interferers, Reported Risk Angles
            # Filter risk angles (remove duplicates or too close to target)
            unique_risks = []
            for ra in risk_angles_all:
                # Check duplication against interferers and target
                is_new = True
                # Check dist to target
                if abs(ra[0] - t_az) < 5 and abs(ra[1] - t_el) < 5:
                    is_new = False
                if is_new:
                    unique_risks.append(ra)

            # Limit number of constraints to avoid over-constraining
            # Max constraints < Total Elements? No, baseband has M degrees of freedom (4).
            # If we add too many constraints, pinv will find min-error solution.
            # Let's prioritize Explicit Interferers + Top 2 Risk Angles
            constraints_list = interferers + unique_risks[:2]

            baseband.optimize_weights(chains, target, constraints_list)

            # 5. System Analysis
            sys_metrics = system.analyze_system_performance(t_az, t_el, constraints_list)

            # 6. Save Data
            record = {
                "id": sample_idx,
                "target": target,
                "interferers": interferers,
                "rf_metrics": rf_metrics_list, # Initial State
                "baseband_weights": [str(w) for w in baseband.weights], # Action
                "system_metrics": sys_metrics, # Reward/Result
                "constraints_used": constraints_list
            }

            f.write(json.dumps(record) + "\n")

            if (sample_idx + 1) % 5 == 0:
                print(f"Generated {sample_idx + 1}/{num_samples}")

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
