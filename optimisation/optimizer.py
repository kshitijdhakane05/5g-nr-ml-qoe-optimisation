"""Offline ML-based network configuration optimiser using Phase 1 QoE surrogate model.

Follows a transparent grid-search approach:
1. Generates candidate configurations within the verified training data support.
2. Applies the exact Phase 1 feature transformations.
3. Predicts QoE using models/qoe_model.joblib.
4. Checks user-specified constraints (e.g., max_power_dbm, min_predicted_qoe).
5. Ranks candidates and compares against a baseline configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

from ml.predict import format_input_features, load_model, predict_qoe
from optimisation.search_space import (
    SUPPORTED_SEARCH_SPACE,
    generate_candidate_configurations,
    validate_network_context,
)


class NetworkOptimiser:
    """Offline grid-search network optimiser based on verified ML surrogate."""

    def __init__(self, model_path: str | Path = "models/qoe_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.model_bundle = load_model(self.model_path)
        self.feature_columns = self.model_bundle["feature_columns"]

    def evaluate_configuration(self, config: dict[str, Any]) -> float:
        """Predict QoE for a single configuration dictionary."""
        features_df = format_input_features(
            ue_count=int(config["ue_count"]),
            bandwidth_mhz=float(config["bandwidth_mhz"]),
            total_tx_power_dbm=float(config["total_tx_power_dbm"]),
            scenario_size_m=float(config.get("scenario_size_m", 30.0)),
            flow_type=str(config.get("flow_type", "voice_be")),
            lambda_val=float(config.get("lambda_val", 10000.0)),
        )
        pred = float(predict_qoe(self.model_bundle, features_df)[0])
        return pred

    def check_constraints(
        self,
        config: dict[str, Any],
        predicted_qoe: float,
        constraints: dict[str, Any] | None,
    ) -> tuple[bool, list[str]]:
        """Check user-provided constraints on candidate configuration.
        
        Supported constraints:
            - max_power_dbm: float (total_tx_power_dbm <= max_power_dbm)
            - min_predicted_qoe: float (predicted_qoe >= min_predicted_qoe)
            - allowed_bandwidths: list[float] (bandwidth_mhz in allowed_bandwidths)
        """
        if not constraints:
            return True, []

        violations = []

        if "max_power_dbm" in constraints and constraints["max_power_dbm"] is not None:
            max_pwr = float(constraints["max_power_dbm"])
            if config["total_tx_power_dbm"] > max_pwr:
                violations.append(
                    f"power ({config['total_tx_power_dbm']} dBm) exceeds max ({max_pwr} dBm)"
                )

        if "min_predicted_qoe" in constraints and constraints["min_predicted_qoe"] is not None:
            min_qoe = float(constraints["min_predicted_qoe"])
            if predicted_qoe < min_qoe:
                violations.append(
                    f"predicted QoE ({predicted_qoe:.4f}) below minimum ({min_qoe:.4f})"
                )

        if "allowed_bandwidths" in constraints and constraints["allowed_bandwidths"] is not None:
            allowed_bws = [float(b) for b in constraints["allowed_bandwidths"]]
            if config["bandwidth_mhz"] not in allowed_bws:
                violations.append(
                    f"bandwidth ({config['bandwidth_mhz']} MHz) not in allowed set {allowed_bws}"
                )

        satisfied = (len(violations) == 0)
        return satisfied, violations

    def optimise(
        self,
        ue_count: int,
        scenario_size_m: float = 30.0,
        flow_type: str = "voice_be",
        lambda_val: float = 10000.0,
        baseline_bandwidth_mhz: float = 20.0,
        baseline_power_dbm: float = 35.0,
        constraints: dict[str, Any] | None = None,
        candidate_bandwidths: list[float] | None = None,
    ) -> dict[str, Any]:
        """Perform offline grid-search optimisation.
        
        Returns dictionary containing:
            - context
            - baseline configuration & predicted QoE
            - best configuration & predicted QoE
            - predicted improvement (pending ns-3 verification)
            - candidate evaluation ranking table
        """
        valid, msg = validate_network_context(ue_count, scenario_size_m, flow_type, lambda_val)
        if not valid:
            raise ValueError(f"Optimisation aborted: {msg}")

        # 1. Baseline Evaluation
        baseline_config = {
            "ue_count": ue_count,
            "scenario_size_m": float(scenario_size_m),
            "flow_type": flow_type.lower(),
            "lambda_val": float(lambda_val),
            "bandwidth_mhz": float(baseline_bandwidth_mhz),
            "total_tx_power_dbm": float(baseline_power_dbm),
        }
        baseline_qoe = self.evaluate_configuration(baseline_config)
        baseline_satisfied, baseline_violations = self.check_constraints(
            baseline_config, baseline_qoe, constraints
        )
        baseline_config["predicted_qoe"] = baseline_qoe
        baseline_config["feasible"] = baseline_satisfied
        baseline_config["violations"] = baseline_violations

        # 2. Generate and Evaluate Candidates
        candidates = generate_candidate_configurations(
            ue_count=ue_count,
            scenario_size_m=scenario_size_m,
            flow_type=flow_type,
            lambda_val=lambda_val,
            candidate_bandwidths=candidate_bandwidths,
        )

        evaluated_records = []
        for cand in candidates:
            pred_qoe = self.evaluate_configuration(cand)
            feasible, violations = self.check_constraints(cand, pred_qoe, constraints)
            record = dict(cand)
            record["predicted_qoe"] = pred_qoe
            record["feasible"] = feasible
            record["violations"] = violations
            evaluated_records.append(record)

        # 3. Rank Candidates: Feasible first, then highest predicted QoE
        ranked_records = sorted(
            evaluated_records,
            key=lambda r: (1 if r["feasible"] else 0, r["predicted_qoe"]),
            reverse=True,
        )

        for rank, r in enumerate(ranked_records, start=1):
            r["rank"] = rank

        # 4. Best Selection
        feasible_records = [r for r in ranked_records if r["feasible"]]
        if feasible_records:
            best_candidate = feasible_records[0]
            selection_status = "FEASIBLE_OPTIMAL"
        else:
            best_candidate = ranked_records[0]
            selection_status = "NO_FEASIBLE_CANDIDATE_HIGHEST_QOE_RETURNED"

        predicted_improvement = best_candidate["predicted_qoe"] - baseline_qoe

        return {
            "context": {
                "ue_count": ue_count,
                "scenario_size_m": scenario_size_m,
                "flow_type": flow_type,
                "lambda_val": lambda_val,
            },
            "constraints": constraints or {},
            "baseline": baseline_config,
            "best_configuration": best_candidate,
            "selection_status": selection_status,
            "predicted_improvement_label": "Predicted improvement — pending ns-3 verification",
            "predicted_qoe_improvement": predicted_improvement,
            "num_candidates_evaluated": len(ranked_records),
            "all_candidates": ranked_records,
        }

