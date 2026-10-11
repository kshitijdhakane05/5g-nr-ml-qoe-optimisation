"""Closed-loop verification workflow integrating Phase 2 optimiser and ns-3 simulation.

Workflow:
1. Accept network context.
2. Query Phase 2 NetworkOptimiser to obtain candidate recommendation & predicted QoE.
3. Launch ns-3 5G simulation for the recommended configuration using Ns3Runner.
4. Optionally launch ns-3 5G simulation for the baseline configuration.
5. Parse actual measured QoS metrics via result_parser.
6. Calculate measured QoE using verified project formula.
7. Compute error metrics:
   - absolute_prediction_error: |predicted_qoe - measured_qoe|
   - relative_prediction_error: |predicted_qoe - measured_qoe| / measured_qoe (when measured_qoe > 0)
   - measured_improvement: measured_qoe_opt - measured_qoe_base
   - predicted_improvement: predicted_qoe_opt - predicted_qoe_base
8. Persist verification records to results/phase3/verification_results.csv.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from optimisation.optimizer import NetworkOptimiser
from simulation.ns3_runner import Ns3RunResult, Ns3Runner
from simulation.result_parser import SimulationMetrics, parse_simulation_csv


@dataclass
class ClosedLoopVerificationRecord:
    run_id: str
    timestamp: str
    # Context
    ue_count: int
    scenario_size_m: float
    flow_type: str
    lambda_val: float
    # Recommended / Optimised Config & Prediction
    optimised_bandwidth_mhz: float
    optimised_power_dbm: float
    optimiser_status: str
    predicted_qoe: float
    # Simulation Execution
    sim_success: bool
    sim_duration_seconds: float
    sim_exit_code: int
    is_outage: bool
    # Measured QoS KPIs
    measured_throughput_mbps: float
    measured_delay_ms: float
    measured_jitter_ms: float
    measured_packet_loss: float
    total_tx_packets: int
    total_rx_packets: int
    # Measured QoE & Verification Error
    measured_qoe: float
    absolute_prediction_error: float | None
    relative_prediction_error: float | None
    # Baseline Comparison
    run_baseline: bool
    baseline_bandwidth_mhz: float | None
    baseline_power_dbm: float | None
    baseline_predicted_qoe: float | None
    baseline_measured_qoe: float | None
    predicted_improvement: float | None
    measured_improvement: float | None
    # Artifact Paths & Notes
    run_dir: str
    csv_path: str
    log_path: str
    verification_status: str
    error_message: str | None = None


@dataclass
class QoERecoveryRecord:
    run_id: str
    timestamp: str
    target_qoe: float
    recovery_attempted: bool
    status: str  # "NOT_NEEDED", "SUCCESS", "TARGET_NOT_MET", "NO_ALTERNATIVE_CONFIGURATION", "SIMULATION_FAILED"
    rationale: str
    previous_bandwidth_mhz: float
    corrected_bandwidth_mhz: float | None
    fixed_parameters: dict[str, Any]
    initial_metrics: dict[str, Any]
    corrected_metrics: dict[str, Any] | None
    target_achieved: bool
    initial_record: ClosedLoopVerificationRecord
    corrective_record: ClosedLoopVerificationRecord | None = None
    error_message: str | None = None



class ClosedLoopVerifier:
    """Orchestrates closed-loop ML optimisation and physical ns-3 simulation."""

    def __init__(
        self,
        model_path: str | Path = "models/qoe_model.joblib",
        output_base_dir: str | Path = "results/phase3/runs",
        results_csv_path: str | Path = "results/phase3/verification_results.csv",
        runner: Ns3Runner | None = None,
    ) -> None:
        self.optimiser = NetworkOptimiser(model_path=model_path)
        self.runner = runner if runner is not None else Ns3Runner(default_output_base=output_base_dir)
        self.results_csv_path = Path(results_csv_path)

    def run_closed_loop_verification(
        self,
        ue_count: int,
        scenario_size_m: float = 30.0,
        flow_type: str = "voice_be",
        lambda_val: float = 10000.0,
        baseline_bandwidth_mhz: float = 20.0,
        baseline_power_dbm: float = 35.0,
        constraints: dict[str, Any] | None = None,
        run_baseline_sim: bool = True,
        sim_time_str: str = "1s",
    ) -> ClosedLoopVerificationRecord:
        """Execute end-to-end closed-loop verification for a given network context."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"cl_{timestamp_str}_ue{ue_count}"

        # 1. Optimiser selection
        opt_res = self.optimiser.optimise(
            ue_count=ue_count,
            scenario_size_m=scenario_size_m,
            flow_type=flow_type,
            lambda_val=lambda_val,
            baseline_bandwidth_mhz=baseline_bandwidth_mhz,
            baseline_power_dbm=baseline_power_dbm,
            constraints=constraints,
        )

        best_cand = opt_res["best_configuration"]
        base_cand = opt_res["baseline"]
        pred_qoe = float(best_cand["predicted_qoe"])
        pred_base_qoe = float(base_cand["predicted_qoe"])
        pred_improvement = opt_res["predicted_qoe_improvement"]

        # 2. Run simulation for the selected optimal configuration
        sim_config_opt = {
            "ue_count": ue_count,
            "scenario_size_m": scenario_size_m,
            "bandwidth_mhz": best_cand["bandwidth_mhz"],
            "total_tx_power_dbm": best_cand["total_tx_power_dbm"],
            "lambda_be": lambda_val,
            "scenario_id": 1,
            "sim_time": sim_time_str,
        }

        opt_run_id = f"{run_id}_opt"
        run_result_opt = self.runner.run_simulation(config=sim_config_opt, run_id=opt_run_id)

        # 3. Parse simulation metrics for the selected configuration
        if run_result_opt.success and run_result_opt.csv_path.exists():
            metrics_opt = parse_simulation_csv(
                run_result_opt.csv_path,
                target_flow_type=flow_type,
            )
        else:
            metrics_opt = SimulationMetrics(
                success=False,
                is_outage=False,
                num_flows=0,
                target_flow_type=flow_type,
                target_flow_count=0,
                mean_throughput_mbps=0.0,
                mean_delay_ms=0.0,
                mean_jitter_ms=0.0,
                mean_packet_loss=0.0,
                mean_qoe_proxy=0.0,
                total_tx_packets=0,
                total_rx_packets=0,
                error_message=run_result_opt.error_message or "Optimised simulation failed",
            )

        # 4. Optional baseline simulation
        base_measured_qoe: float | None = None
        meas_improvement: float | None = None

        if run_baseline_sim:
            sim_config_base = {
                "ue_count": ue_count,
                "scenario_size_m": scenario_size_m,
                "bandwidth_mhz": base_cand["bandwidth_mhz"],
                "total_tx_power_dbm": base_cand["total_tx_power_dbm"],
                "lambda_be": lambda_val,
                "scenario_id": 1,
                "sim_time": sim_time_str,
            }
            base_run_id = f"{run_id}_base"
            run_result_base = self.runner.run_simulation(config=sim_config_base, run_id=base_run_id)

            if run_result_base.success and run_result_base.csv_path.exists():
                metrics_base = parse_simulation_csv(
                    run_result_base.csv_path,
                    target_flow_type=flow_type,
                )
                if metrics_base.success:
                    base_measured_qoe = metrics_base.mean_qoe_proxy

        # 5. Error calculations
        if metrics_opt.success:
            meas_qoe = metrics_opt.mean_qoe_proxy
            abs_err = abs(pred_qoe - meas_qoe)
            # Relative error is mathematically meaningful only when measured_qoe > 0
            rel_err = (abs_err / meas_qoe) if meas_qoe > 1e-6 else None

            if base_measured_qoe is not None:
                meas_improvement = meas_qoe - base_measured_qoe

            verif_status = "VERIFIED_SUCCESS" if abs_err <= 0.15 else "VERIFIED_HIGH_ERROR"
        else:
            meas_qoe = 0.0
            abs_err = None
            rel_err = None
            verif_status = "SIMULATION_FAILED"

        record = ClosedLoopVerificationRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ue_count=ue_count,
            scenario_size_m=scenario_size_m,
            flow_type=flow_type,
            lambda_val=lambda_val,
            optimised_bandwidth_mhz=best_cand["bandwidth_mhz"],
            optimised_power_dbm=best_cand["total_tx_power_dbm"],
            optimiser_status=opt_res["selection_status"],
            predicted_qoe=pred_qoe,
            sim_success=run_result_opt.success,
            sim_duration_seconds=run_result_opt.duration_seconds,
            sim_exit_code=run_result_opt.exit_code,
            is_outage=metrics_opt.is_outage,
            measured_throughput_mbps=metrics_opt.mean_throughput_mbps,
            measured_delay_ms=metrics_opt.mean_delay_ms,
            measured_jitter_ms=metrics_opt.mean_jitter_ms,
            measured_packet_loss=metrics_opt.mean_packet_loss,
            total_tx_packets=metrics_opt.total_tx_packets,
            total_rx_packets=metrics_opt.total_rx_packets,
            measured_qoe=meas_qoe,
            absolute_prediction_error=abs_err,
            relative_prediction_error=rel_err,
            run_baseline=run_baseline_sim,
            baseline_bandwidth_mhz=base_cand["bandwidth_mhz"],
            baseline_power_dbm=base_cand["total_tx_power_dbm"],
            baseline_predicted_qoe=pred_base_qoe,
            baseline_measured_qoe=base_measured_qoe,
            predicted_improvement=pred_improvement,
            measured_improvement=meas_improvement,
            run_dir=str(run_result_opt.run_dir),
            csv_path=str(run_result_opt.csv_path),
            log_path=str(run_result_opt.run_dir / "stdout.log"),
            verification_status=verif_status,
            error_message=metrics_opt.error_message,
        )

        self.persist_record(record)
        return record

    def run_single_simulation(
        self,
        ue_count: int,
        bandwidth_mhz: float,
        total_tx_power_dbm: float = 35.0,
        scenario_size_m: float = 30.0,
        flow_type: str = "voice_be",
        lambda_val: float = 10000.0,
        sim_time_str: str = "1s",
        run_tag: str = "single",
    ) -> ClosedLoopVerificationRecord:
        """Run an ns-3 simulation for an explicit configuration without baseline comparison."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"sim_{timestamp_str}_bw{int(bandwidth_mhz)}_{run_tag}"

        # Surrogate prediction for the explicit configuration
        pred_qoe = self.optimiser.evaluate_configuration({
            "ue_count": ue_count,
            "bandwidth_mhz": bandwidth_mhz,
            "total_tx_power_dbm": total_tx_power_dbm,
            "scenario_size_m": scenario_size_m,
            "flow_type": flow_type,
            "lambda_val": lambda_val,
        })

        sim_config = {
            "ue_count": ue_count,
            "scenario_size_m": scenario_size_m,
            "bandwidth_mhz": bandwidth_mhz,
            "total_tx_power_dbm": total_tx_power_dbm,
            "lambda_be": lambda_val,
            "scenario_id": 1,
            "sim_time": sim_time_str,
        }

        run_result = self.runner.run_simulation(config=sim_config, run_id=run_id)

        if run_result.success and run_result.csv_path.exists():
            metrics = parse_simulation_csv(
                run_result.csv_path,
                target_flow_type=flow_type,
            )
        else:
            metrics = SimulationMetrics(
                success=False,
                is_outage=False,
                num_flows=0,
                target_flow_type=flow_type,
                target_flow_count=0,
                mean_throughput_mbps=0.0,
                mean_delay_ms=0.0,
                mean_jitter_ms=0.0,
                mean_packet_loss=0.0,
                mean_qoe_proxy=0.0,
                total_tx_packets=0,
                total_rx_packets=0,
                error_message=run_result.error_message or "Simulation failed",
            )

        if metrics.success:
            meas_qoe = metrics.mean_qoe_proxy
            abs_err = abs(pred_qoe - meas_qoe)
            rel_err = (abs_err / meas_qoe) if meas_qoe > 1e-6 else None
            verif_status = "VERIFIED_SUCCESS" if abs_err <= 0.15 else "VERIFIED_HIGH_ERROR"
        else:
            meas_qoe = 0.0
            abs_err = None
            rel_err = None
            verif_status = "SIMULATION_FAILED"

        record = ClosedLoopVerificationRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ue_count=ue_count,
            scenario_size_m=scenario_size_m,
            flow_type=flow_type,
            lambda_val=lambda_val,
            optimised_bandwidth_mhz=bandwidth_mhz,
            optimised_power_dbm=total_tx_power_dbm,
            optimiser_status="DIRECT_CONFIGURATION",
            predicted_qoe=pred_qoe,
            sim_success=run_result.success,
            sim_duration_seconds=run_result.duration_seconds,
            sim_exit_code=run_result.exit_code,
            is_outage=metrics.is_outage,
            measured_throughput_mbps=metrics.mean_throughput_mbps,
            measured_delay_ms=metrics.mean_delay_ms,
            measured_jitter_ms=metrics.mean_jitter_ms,
            measured_packet_loss=metrics.mean_packet_loss,
            total_tx_packets=metrics.total_tx_packets,
            total_rx_packets=metrics.total_rx_packets,
            measured_qoe=meas_qoe,
            absolute_prediction_error=abs_err,
            relative_prediction_error=rel_err,
            run_baseline=False,
            baseline_bandwidth_mhz=None,
            baseline_power_dbm=None,
            baseline_predicted_qoe=None,
            baseline_measured_qoe=None,
            predicted_improvement=None,
            measured_improvement=None,
            run_dir=str(run_result.run_dir),
            csv_path=str(run_result.csv_path),
            log_path=str(run_result.run_dir / "stdout.log"),
            verification_status=verif_status,
            error_message=metrics.error_message,
        )
        self.persist_record(record)
        return record

    def run_verification_with_recovery(
        self,
        ue_count: int,
        initial_bandwidth_mhz: float = 20.0,
        total_tx_power_dbm: float = 35.0,
        scenario_size_m: float = 30.0,
        flow_type: str = "voice_be",
        lambda_val: float = 10000.0,
        target_qoe: float = 0.80,
        sim_time_str: str = "1s",
        recovery_results_csv_path: str | Path = "results/phase3/recovery_results.csv",
    ) -> QoERecoveryRecord:
        """Run an initial real ns-3 simulation and perform at most one corrective simulation if needed.
        
        Rules:
        - If initial measured QoE >= target_qoe: NOT_NEEDED.
        - If initial measured QoE < target_qoe:
            - If initial_bandwidth_mhz == 20.0: Corrective bandwidth is 50.0 MHz.
            - If initial_bandwidth_mhz == 50.0: NO_ALTERNATIVE_CONFIGURATION (no higher supported BW).
        - Limit: Exactly at most 1 corrective simulation attempt.
        - Physical ns-3 simulation executed for corrective attempt (no ML substitution).
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        recovery_run_id = f"recovery_{timestamp_str}_ue{ue_count}"

        fixed_params = {
            "ue_count": ue_count,
            "total_tx_power_dbm": total_tx_power_dbm,
            "scenario_size_m": scenario_size_m,
            "flow_type": flow_type,
            "lambda_val": lambda_val,
            "sim_time": sim_time_str,
        }

        # Step 1: Run initial simulation
        initial_record = self.run_single_simulation(
            ue_count=ue_count,
            bandwidth_mhz=initial_bandwidth_mhz,
            total_tx_power_dbm=total_tx_power_dbm,
            scenario_size_m=scenario_size_m,
            flow_type=flow_type,
            lambda_val=lambda_val,
            sim_time_str=sim_time_str,
            run_tag="initial",
        )

        initial_metrics = {
            "measured_qoe": initial_record.measured_qoe,
            "measured_throughput_mbps": initial_record.measured_throughput_mbps,
            "measured_delay_ms": initial_record.measured_delay_ms,
            "measured_jitter_ms": initial_record.measured_jitter_ms,
            "measured_packet_loss": initial_record.measured_packet_loss,
            "is_outage": initial_record.is_outage,
            "sim_success": initial_record.sim_success,
        }

        # Case 1: Initial simulation crashed or failed execution
        if not initial_record.sim_success:
            rec = QoERecoveryRecord(
                run_id=recovery_run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                target_qoe=target_qoe,
                recovery_attempted=False,
                status="SIMULATION_FAILED",
                rationale="Initial ns-3 simulation failed to execute successfully.",
                previous_bandwidth_mhz=initial_bandwidth_mhz,
                corrected_bandwidth_mhz=None,
                fixed_parameters=fixed_params,
                initial_metrics=initial_metrics,
                corrected_metrics=None,
                target_achieved=False,
                initial_record=initial_record,
                corrective_record=None,
                error_message=initial_record.error_message or "Initial simulation failed",
            )
            self.persist_recovery_record(rec, recovery_results_csv_path)
            return rec

        # Case 2: Initial measured QoE already meets or exceeds target
        if initial_record.measured_qoe >= target_qoe:
            rec = QoERecoveryRecord(
                run_id=recovery_run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                target_qoe=target_qoe,
                recovery_attempted=False,
                status="NOT_NEEDED",
                rationale=(
                    f"Initial measured QoE ({initial_record.measured_qoe:.4f}) already meets "
                    f"or exceeds target QoE ({target_qoe:.4f})."
                ),
                previous_bandwidth_mhz=initial_bandwidth_mhz,
                corrected_bandwidth_mhz=None,
                fixed_parameters=fixed_params,
                initial_metrics=initial_metrics,
                corrected_metrics=None,
                target_achieved=True,
                initial_record=initial_record,
                corrective_record=None,
                error_message=None,
            )
            self.persist_recovery_record(rec, recovery_results_csv_path)
            return rec

        # Initial measured QoE is below target: determine corrective action
        if initial_bandwidth_mhz >= 50.0:
            rec = QoERecoveryRecord(
                run_id=recovery_run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                target_qoe=target_qoe,
                recovery_attempted=False,
                status="NO_ALTERNATIVE_CONFIGURATION",
                rationale=(
                    f"Initial measured QoE ({initial_record.measured_qoe:.4f}) is below target "
                    f"({target_qoe:.4f}), but network is already at maximum supported bandwidth "
                    f"(50.0 MHz). No higher supported bandwidth alternative exists in {{20.0, 50.0}} MHz."
                ),
                previous_bandwidth_mhz=initial_bandwidth_mhz,
                corrected_bandwidth_mhz=None,
                fixed_parameters=fixed_params,
                initial_metrics=initial_metrics,
                corrected_metrics=None,
                target_achieved=False,
                initial_record=initial_record,
                corrective_record=None,
                error_message=None,
            )
            self.persist_recovery_record(rec, recovery_results_csv_path)
            return rec

        # Case 3: Initial bandwidth is 20 MHz -> Corrective bandwidth is 50 MHz
        corrected_bw = 50.0
        rationale = (
            f"Initial measured QoE ({initial_record.measured_qoe:.4f}) fell below target "
            f"({target_qoe:.4f}). Upgraded bandwidth from {initial_bandwidth_mhz:.1f} MHz to "
            f"{corrected_bw:.1f} MHz while maintaining Tx power at {total_tx_power_dbm:.1f} dBm."
        )

        # Run at most one corrective simulation
        corrective_record = self.run_single_simulation(
            ue_count=ue_count,
            bandwidth_mhz=corrected_bw,
            total_tx_power_dbm=total_tx_power_dbm,
            scenario_size_m=scenario_size_m,
            flow_type=flow_type,
            lambda_val=lambda_val,
            sim_time_str=sim_time_str,
            run_tag="corrective",
        )

        corrected_metrics = {
            "measured_qoe": corrective_record.measured_qoe,
            "measured_throughput_mbps": corrective_record.measured_throughput_mbps,
            "measured_delay_ms": corrective_record.measured_delay_ms,
            "measured_jitter_ms": corrective_record.measured_jitter_ms,
            "measured_packet_loss": corrective_record.measured_packet_loss,
            "is_outage": corrective_record.is_outage,
            "sim_success": corrective_record.sim_success,
        }

        if not corrective_record.sim_success:
            rec = QoERecoveryRecord(
                run_id=recovery_run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                target_qoe=target_qoe,
                recovery_attempted=True,
                status="SIMULATION_FAILED",
                rationale=rationale,
                previous_bandwidth_mhz=initial_bandwidth_mhz,
                corrected_bandwidth_mhz=corrected_bw,
                fixed_parameters=fixed_params,
                initial_metrics=initial_metrics,
                corrected_metrics=corrected_metrics,
                target_achieved=False,
                initial_record=initial_record,
                corrective_record=corrective_record,
                error_message=corrective_record.error_message or "Corrective simulation failed",
            )
            self.persist_recovery_record(rec, recovery_results_csv_path)
            return rec

        target_achieved = corrective_record.measured_qoe >= target_qoe
        recovery_status = "SUCCESS" if target_achieved else "TARGET_NOT_MET"

        rec = QoERecoveryRecord(
            run_id=recovery_run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_qoe=target_qoe,
            recovery_attempted=True,
            status=recovery_status,
            rationale=rationale,
            previous_bandwidth_mhz=initial_bandwidth_mhz,
            corrected_bandwidth_mhz=corrected_bw,
            fixed_parameters=fixed_params,
            initial_metrics=initial_metrics,
            corrected_metrics=corrected_metrics,
            target_achieved=target_achieved,
            initial_record=initial_record,
            corrective_record=corrective_record,
            error_message=None,
        )
        self.persist_recovery_record(rec, recovery_results_csv_path)
        return rec

    def persist_record(self, record: ClosedLoopVerificationRecord) -> None:
        """Append record to results CSV."""
        self.results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        row_dict = asdict(record)
        df = pd.DataFrame([row_dict])
        if self.results_csv_path.exists():
            df.to_csv(self.results_csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.results_csv_path, index=False)

    def persist_recovery_record(
        self,
        record: QoERecoveryRecord,
        csv_path: str | Path,
    ) -> None:
        """Append recovery summary record to recovery results CSV."""
        target_path = Path(csv_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        summary_row = {
            "run_id": record.run_id,
            "timestamp": record.timestamp,
            "target_qoe": record.target_qoe,
            "recovery_attempted": record.recovery_attempted,
            "status": record.status,
            "previous_bandwidth_mhz": record.previous_bandwidth_mhz,
            "corrected_bandwidth_mhz": record.corrected_bandwidth_mhz,
            "target_achieved": record.target_achieved,
            "initial_measured_qoe": record.initial_metrics.get("measured_qoe", 0.0),
            "corrected_measured_qoe": (
                record.corrected_metrics.get("measured_qoe") if record.corrected_metrics else None
            ),
            "initial_throughput_mbps": record.initial_metrics.get("measured_throughput_mbps", 0.0),
            "corrected_throughput_mbps": (
                record.corrected_metrics.get("measured_throughput_mbps") if record.corrected_metrics else None
            ),
            "initial_delay_ms": record.initial_metrics.get("measured_delay_ms", 0.0),
            "corrected_delay_ms": (
                record.corrected_metrics.get("measured_delay_ms") if record.corrected_metrics else None
            ),
            "initial_jitter_ms": record.initial_metrics.get("measured_jitter_ms", 0.0),
            "corrected_jitter_ms": (
                record.corrected_metrics.get("measured_jitter_ms") if record.corrected_metrics else None
            ),
            "initial_packet_loss": record.initial_metrics.get("measured_packet_loss", 0.0),
            "corrected_packet_loss": (
                record.corrected_metrics.get("measured_packet_loss") if record.corrected_metrics else None
            ),
            "rationale": record.rationale,
            "error_message": record.error_message,
        }
        df = pd.DataFrame([summary_row])
        if target_path.exists():
            df.to_csv(target_path, mode="a", header=False, index=False)
        else:
            df.to_csv(target_path, index=False)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Closed-Loop ns-3 Simulation Verification"
    )
    parser.add_argument("--ue-count", type=int, default=2, help="Number of active UEs (2, 4, 6, 8, 10)")
    parser.add_argument("--scenario-size", type=float, default=3.0, help="Area size in meters (3, 30, 100)")
    parser.add_argument(
        "--flow-type",
        choices=["voice_be", "low_latency"],
        default="voice_be",
        help="Traffic class",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=5000.0,
        help="Packet arrival rate (5000, 10000, 20000, 40000)",
    )
    parser.add_argument(
        "--baseline-bandwidth",
        type=float,
        default=20.0,
        help="Baseline bandwidth in MHz (default: 20.0)",
    )
    parser.add_argument(
        "--baseline-power",
        type=float,
        default=35.0,
        help="Baseline power in dBm (default: 35.0)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip running the baseline simulation",
    )
    parser.add_argument(
        "--sim-time",
        default="1s",
        help="Simulation duration string (default: 1s)",
    )
    parser.add_argument(
        "--model-path",
        default="models/qoe_model.joblib",
        help="Path to Phase 1 trained model",
    )
    parser.add_argument(
        "--results-csv",
        default="results/phase3/verification_results.csv",
        help="Path to verification results CSV",
    )
    parser.add_argument(
        "--enable-recovery",
        action="store_true",
        help="Run QoE Recovery Mode if measured QoE is below target",
    )
    parser.add_argument(
        "--target-qoe",
        type=float,
        default=0.80,
        help="Target QoE threshold for recovery mode (default: 0.80)",
    )
    parser.add_argument(
        "--recovery-csv",
        default="results/phase3/recovery_results.csv",
        help="Path to recovery results CSV",
    )
    args = parser.parse_args()

    verifier = ClosedLoopVerifier(
        model_path=args.model_path,
        results_csv_path=args.results_csv,
    )

    print("\n" + "=" * 70)
    print("PHASE 3: CLOSED-LOOP NS-3 5G VERIFICATION")
    print("Mode: Offline Simulation-Based Verification (NOT Live Control)")
    if args.enable_recovery:
        print(f"Feature: QoE Recovery Mode (Target QoE: {args.target_qoe:.2f})")
    print("=" * 70)

    print("\n[1] NETWORK CONTEXT:")
    print(f"  UE Count:            {args.ue_count}")
    print(f"  Area Size:           {args.scenario_size} m")
    print(f"  Flow Type:           {args.flow_type}")
    print(f"  Packet Rate:         {args.lambda_val} pkts/s")

    if args.enable_recovery:
        print(f"  Initial Bandwidth:   {args.baseline_bandwidth} MHz")
        recovery_rec = verifier.run_verification_with_recovery(
            ue_count=args.ue_count,
            initial_bandwidth_mhz=args.baseline_bandwidth,
            total_tx_power_dbm=args.baseline_power,
            scenario_size_m=args.scenario_size,
            flow_type=args.flow_type,
            lambda_val=args.lambda_val,
            target_qoe=args.target_qoe,
            sim_time_str=args.sim_time,
            recovery_results_csv_path=args.recovery_csv,
        )

        print("\n[2] QOE RECOVERY EXECUTION & DIAGNOSTICS:")
        print(f"  Recovery Status:     {recovery_rec.status}")
        print(f"  Recovery Attempted:  {recovery_rec.recovery_attempted}")
        print(f"  Target Achieved:     {recovery_rec.target_achieved}")
        print(f"  Rationale:           {recovery_rec.rationale}")
        print(f"  Initial BW:          {recovery_rec.previous_bandwidth_mhz} MHz")
        if recovery_rec.corrected_bandwidth_mhz is not None:
            print(f"  Corrected BW:        {recovery_rec.corrected_bandwidth_mhz} MHz")

        print("\n[3] MEASURED METRICS COMPARISON:")
        print(f"  Initial Measured QoE:   {recovery_rec.initial_metrics.get('measured_qoe', 0.0):.4f}")
        if recovery_rec.corrected_metrics:
            print(f"  Corrected Measured QoE: {recovery_rec.corrected_metrics.get('measured_qoe', 0.0):.4f}")
            print(f"  Throughput (Initial -> Corrected): {recovery_rec.initial_metrics.get('measured_throughput_mbps', 0.0):.2f} -> {recovery_rec.corrected_metrics.get('measured_throughput_mbps', 0.0):.2f} Mbps")
            print(f"  Delay (Initial -> Corrected):      {recovery_rec.initial_metrics.get('measured_delay_ms', 0.0):.2f} -> {recovery_rec.corrected_metrics.get('measured_delay_ms', 0.0):.2f} ms")
            print(f"  Jitter (Initial -> Corrected):     {recovery_rec.initial_metrics.get('measured_jitter_ms', 0.0):.2f} -> {recovery_rec.corrected_metrics.get('measured_jitter_ms', 0.0):.2f} ms")
            print(f"  Loss (Initial -> Corrected):       {recovery_rec.initial_metrics.get('measured_packet_loss', 0.0):.4f} -> {recovery_rec.corrected_metrics.get('measured_packet_loss', 0.0):.4f}")

        print("\n[4] PERSISTENCE & TRACEABILITY:")
        print(f"  Recovery CSV:        {args.recovery_csv}")
        print(f"  Verification CSV:    {verifier.results_csv_path}")
        print("=" * 70 + "\n")
        return

    rec = verifier.run_closed_loop_verification(
        ue_count=args.ue_count,
        scenario_size_m=args.scenario_size,
        flow_type=args.flow_type,
        lambda_val=args.lambda_val,
        baseline_bandwidth_mhz=args.baseline_bandwidth,
        baseline_power_dbm=args.baseline_power,
        run_baseline_sim=not args.no_baseline,
        sim_time_str=args.sim_time,
    )

    print("\n[2] OPTIMISER SELECTION & PREDICTION:")
    print(f"  Selected Bandwidth:  {rec.optimised_bandwidth_mhz} MHz")
    print(f"  Selected Tx Power:   {rec.optimised_power_dbm} dBm")
    print(f"  Predicted QoE:       {rec.predicted_qoe:.4f}")

    print("\n[3] ACTUAL NS-3 SIMULATION MEASUREMENTS:")
    print(f"  Simulation Status:   {'SUCCESS' if rec.sim_success else 'FAILED'}")
    print(f"  Duration:            {rec.sim_duration_seconds:.2f} s")
    print(f"  Is Outage:           {rec.is_outage}")
    print(f"  Throughput:          {rec.measured_throughput_mbps:.4f} Mbps")
    print(f"  Mean Delay:          {rec.measured_delay_ms:.4f} ms")
    print(f"  Mean Jitter:         {rec.measured_jitter_ms:.4f} ms")
    print(f"  Packet Loss Ratio:   {rec.measured_packet_loss:.4f}")
    print(f"  Measured QoE:        {rec.measured_qoe:.4f}")

    print("\n[4] PREDICTED VS MEASURED COMPARISON:")
    print("-" * 70)
    print(f"  Predicted QoE:       {rec.predicted_qoe:.4f}")
    print(f"  Measured QoE:        {rec.measured_qoe:.4f}")
    if rec.absolute_prediction_error is not None:
        print(f"  Absolute Error:      {rec.absolute_prediction_error:.4f}")
    if rec.relative_prediction_error is not None:
        print(f"  Relative Error:      {rec.relative_prediction_error * 100:.2f}%")
    else:
        print("  Relative Error:      N/A (measured QoE is zero or undefined)")

    if rec.run_baseline and rec.baseline_measured_qoe is not None:
        print("\n[5] MEASURED BASELINE VS MEASURED OPTIMISED GAIN:")
        print("-" * 70)
        print(f"  Baseline Bandwidth:  {rec.baseline_bandwidth_mhz} MHz")
        print(f"  Baseline Meas QoE:   {rec.baseline_measured_qoe:.4f}")
        print(f"  Optimised Meas QoE:  {rec.measured_qoe:.4f}")
        print(f"  Measured QoE Gain:   {rec.measured_improvement:+.4f}")
        print(f"  Predicted QoE Gain:  {rec.predicted_improvement:+.4f}")

    print("\n[6] PERSISTENCE & TRACEABILITY:")
    print(f"  Verification Status: {rec.verification_status}")
    print(f"  Run Directory:       {rec.run_dir}")
    print(f"  Results CSV:         {verifier.results_csv_path}")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    main()

