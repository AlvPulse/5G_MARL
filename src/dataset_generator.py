import numpy as np
import json
import os
from typing import List, Dict

from src.geometry import SparsePlanarPanel
from src.algorithms import IterativePOCSSolver
from src.rf_chain import RFChain
from src.baseband import BasebandProcessor
from src.system import MultiPanelSystem
from src.brain import BeamformingBrain
from src.evaluator import SystemEvaluator

def generate_dataset(
    num_samples: int = 10,
    output_file: str = "dataset.jsonl"
):
    """
    Generates a dataset of beamforming scenarios.
    """

    # 1. System Setup (Static)
    # 8 Panels (Octagon) - User Requirement
    # Angles: 0, 45, 90, 135, 180, 225, 270, 315
    num_panels = 8
    radius = 1.0
    orientations = []
    positions = []

    for i in range(num_panels):
        az = i * (360.0 / num_panels)
        orientations.append((az, 0)) # Horizontal
        # Position on circle
        x = radius * np.cos(np.radians(az))
        y = radius * np.sin(np.radians(az))
        positions.append((x, y, 0))

    # User Requirement: Rectangular 16x4 Arrays
    rows, cols = 16, 4
    freq = 28e9

    # Create Panels
    panels = []
    for i in range(num_panels):
        # 50% Sparsity (More challenging)
        total = rows * cols
        mask_flat = np.ones(total, dtype=bool)
        mask_flat[np.random.choice(total, int(0.5*total), replace=False)] = False
        mask = mask_flat.reshape((rows, cols))

        p = SparsePlanarPanel(rows, cols, freq, mask, positions[i], orientations[i])
        panels.append(p)

    print(f"Generating {num_samples} samples with 8 panels and SINR evaluation...")

    with open(output_file, 'w') as f:
        for sample_idx in range(num_samples):
            # 2. Define Ground Truth Targets (2 to 4 users)
            num_users = np.random.randint(2, 5)
            gt_targets = []
            for _ in range(num_users):
                t_az = np.random.uniform(0, 360)
                t_el = np.random.uniform(5, 45)
                gt_targets.append((t_az, t_el))

            # 3. Create Hypothesized Targets (Ground Truth + Error)
            # Simulate "Brain" estimation error (e.g., +/- 5 degrees)
            hyp_targets = []
            for t in gt_targets:
                err_az = np.random.normal(0, 2.0)
                err_el = np.random.normal(0, 2.0)
                hyp_targets.append((t[0] + err_az, t[1] + err_el))

            # 4. Setup System Components
            chains = []
            for p in panels:
                # Use fewer iterations for speed in dataset gen
                solver = IterativePOCSSolver(max_iterations=20)
                chain = RFChain(p, solver)
                chains.append(chain)

            baseband = BasebandProcessor()
            system = MultiPanelSystem(panels, chains, baseband)
            brain = BeamformingBrain(system)
            evaluator = SystemEvaluator(system)

            # 5. Run Brain Logic (Distribution + Beamforming)
            # This updates the RF chains and calculates baseband weights for each hyp_target
            brain_results = brain.distribute_and_process(hyp_targets, fov_deg=60.0)

            # Extract weights for Evaluator
            weights_dict = {}
            for i, metrics in brain_results.items():
                if 'weights' in metrics:
                    weights_dict[i] = metrics['weights']

            # 6. Evaluate SINR against Ground Truth
            sinr_metrics = evaluator.compute_sinr_metrics(
                gt_targets, weights_dict, hyp_targets, noise_power_db=-100.0
            )

            # 7. Save Data
            # Clean numpy arrays for JSON
            cleaned_brain_results = {}
            for k, v in brain_results.items():
                v_clean = v.copy()
                if 'weights' in v_clean:
                    # Convert complex weights to list of strings
                    v_clean['weights'] = [str(w) for w in v_clean['weights']]
                cleaned_brain_results[k] = v_clean

            record = {
                "id": sample_idx,
                "ground_truth_targets": gt_targets,
                "hypothesized_targets": hyp_targets,
                "brain_metrics": cleaned_brain_results,
                "sinr_evaluation": sinr_metrics
            }

            f.write(json.dumps(record) + "\n")

            if (sample_idx + 1) % 5 == 0:
                print(f"Generated {sample_idx + 1}/{num_samples}")

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
