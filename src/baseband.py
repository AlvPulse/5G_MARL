import numpy as np
from typing import List, Dict

class BasebandProcessor:
    """
    Central Baseband Processor.
    Combines signals from multiple RF chains based on their self-reported quality metrics.
    """

    def process_smart(self, rf_signals: List[complex], beam_metrics_list: List[Dict]) -> complex:
        """
        Combines RF signals using weights derived from beam metrics.

        Args:
            rf_signals: List of complex IQ samples (one per RF chain).
            beam_metrics_list: List of dictionaries from RFChain.analyze_performance().

        Returns:
            Combined complex signal.
        """
        if len(rf_signals) != len(beam_metrics_list):
            raise ValueError("Number of signals must match number of metrics reports.")

        weights = []

        for metrics in beam_metrics_list:
            # Heuristic Purity Score Calculation
            # We want High Purity = High Weight.
            # Good beam: High Null Depth (e.g. 40dB), High SLL suppression (e.g. 20dB).
            # Bad beam: Low Null Depth (e.g. 5dB), Low SLL suppression (e.g. 3dB).

            null_depth = metrics.get("min_null_depth_db", 0.0)
            sll_suppression = metrics.get("peak_sll_db", 0.0)

            # Simple linear combination
            # Note: inputs are "dB difference", so higher is better.
            # Avoid negative weights if metrics are weird, though these should be positive.
            score = null_depth + sll_suppression

            # Ensure non-negative
            score = max(score, 1e-6)

            weights.append(score)

        weights = np.array(weights)

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        # Combine
        combined_signal = np.dot(weights, np.array(rf_signals))

        return combined_signal
