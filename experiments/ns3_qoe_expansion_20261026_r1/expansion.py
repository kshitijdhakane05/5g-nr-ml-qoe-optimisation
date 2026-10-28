"""Run the isolated 22-invocation fractional ns-3 QoE expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import pilot
from .preprocessing import derive_effective_band_powers_dbm, derive_flow_inputs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/experiments/ns3_qoe_expansion_20261026_r1_expansion"
SEED = 20261026
RNG_RUNS = (1, 2)
LAMBDA_BE = 20_000
TIMEOUT_SECONDS = 120
EXPANSION_TABLE = (
    ("L9-01", 20, 30, 2),
    ("L9-02", 20, 35, 6),
    ("L9-03", 20, 40, 10),
    ("L9-04", 50, 30, 6),
    ("L9-05", 50, 35, 10),
    ("L9-06", 50, 40, 2),
    ("L9-07", 100, 30, 10),
    ("L9-08", 100, 35, 2),
    ("L9-09", 100, 40, 6),
    ("ANCHOR-UE2", 50, 35, 2),
    ("ANCHOR-UE6", 50, 35, 6),
)
MANIFEST_COLUMNS = ["config_id", "repetition_id", *pilot.MANIFEST_COLUMNS]


def expansion_configurations() -> list[dict[str, Any]]:
    """Return 22 explicit run records: 11 conditions × RngRun 1 and 2."""
    configs = []
    scenario_id = 910001
    for config_id, band1, power, ues in EXPANSION_TABLE:
        for repetition_id, rng_run in enumerate(RNG_RUNS, start=1):
            run_id = f"{config_id.lower().replace('-', '_')}_rep{repetition_id}_rng{rng_run}"
            configs.append({
                "config_id": config_id,
                "repetition_id": repetition_id,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "bandwidth_band1_mhz": band1,
                "bandwidth_band2_mhz": 50,
                "total_tx_power_dbm": power,
                "ue_count": ues,
                "gNb_count": 1,
                "lambda_be_pps": LAMBDA_BE,
                "lambda_ull_pps": 10_000,
                "packet_size_be_bytes": 1252,
                "packet_size_ull_bytes": 100,
                "scenario_size_m": 30,
                "sim_time": "1s",
                "app_start_seconds": 0.4,
                "numerology_bwp1": 4,
                "numerology_bwp2": 2,
                "central_frequency_band1_hz": 28_000_000_000,
                "central_frequency_band2_hz": 28_200_000_000,
                "rng_seed": SEED,
                "rng_run": rng_run,
            })
            scenario_id += 1
    return configs


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            serialized = {
                **record,
                "command_json": json.dumps(record.get("command", []), separators=(",", ":")),
                "metrics_json": json.dumps(record.get("metrics", {}), separators=(",", ":")),
            }
            writer.writerow({key: serialized.get(key, "") for key in MANIFEST_COLUMNS})


def _record_feature_rows(config: dict[str, Any], csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    derived = []
    for row in rows:
        flow = derive_flow_inputs(
            ue_count=config["ue_count"],
            flow_id=int(row["flow_id"]),
            lambda_be_pps=config["lambda_be_pps"],
            lambda_ull_pps=config["lambda_ull_pps"],
            packet_size_ull_bytes=config["packet_size_ull_bytes"],
            packet_size_be_bytes=config["packet_size_be_bytes"],
        )
        derived.append({
            "config_id": config["config_id"],
            "repetition_id": config["repetition_id"],
            "rng_seed": config["rng_seed"],
            "rng_run": config["rng_run"],
            "run_id": config["run_id"],
            "bandwidth_band2_mhz": config["bandwidth_band2_mhz"],
            "effective_band1_tx_power_dbm": derive_effective_band_powers_dbm(
                config["total_tx_power_dbm"], config["bandwidth_band1_mhz"],
                config["bandwidth_band2_mhz"])[0],
            "effective_band2_tx_power_dbm": derive_effective_band_powers_dbm(
                config["total_tx_power_dbm"], config["bandwidth_band1_mhz"],
                config["bandwidth_band2_mhz"])[1],
            "lambda_ull_pps": config["lambda_ull_pps"],
            "packet_size_be_bytes_configured": config["packet_size_be_bytes"],
            "packet_size_ull_bytes_configured": config["packet_size_ull_bytes"],
            **{key: row[key] for key in pilot.CSV_COLUMNS},
            "flow_type": flow["flow_type"],
            "flow_type_id": 0 if flow["flow_type"] == "low_latency" else 1,
            "flow_lambda": flow["flow_lambda_pps"],
            "packet_size_bytes": flow["packet_size_bytes"],
            "offered_throughput_mbps": flow["offered_throughput_mbps_configured"],
        })
    return derived


def _validate_design_records(records: list[dict[str, Any]], configs: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [r for r in records if r.get("status") == "PASS"]
    unique_run_ids = len({r["run_id"] for r in records}) == len(records)
    unique_scenario_ids = len({r["scenario_id"] for r in records}) == len(records)
    by_config: dict[str, list[int]] = {}
    for record in passed:
        by_config.setdefault(record["config_id"], []).append(int(record["rng_run"]))
    paired = len(by_config) == len(EXPANSION_TABLE) and all(
        sorted(run_values) == [1, 2] for run_values in by_config.values()
    )
    all_passed = len(passed) == 22 and len(records) == 22
    return {
        "design_run_count": len(configs),
        "attempted_run_count": len(records),
        "success_count": len(passed),
        "failure_count": sum(r.get("status") != "PASS" for r in records),
        "retry_count": sum(int(r.get("attempt_number", 1)) > 1 for r in records),
        "all_22_designated_runs_passed": all_passed,
        "unique_run_ids": unique_run_ids,
        "unique_scenario_ids": unique_scenario_ids,
        "paired_rng_runs_1_2_per_configuration": paired,
        "all_runs_pass": all_passed and unique_run_ids and unique_scenario_ids and paired,
    }


def run_expansion(output_root: Path, binary: Path, timeout_seconds: float = TIMEOUT_SECONDS) -> int:
    output_root = output_root.resolve()
    binary = binary.resolve()
    source_path = pilot.SOURCE_FILE
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse existing expansion output: {output_root}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"ns-3 executable is absent or not executable: {binary}")
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive")

    # Verify the compiled command-line contract before starting a campaign.
    globals_help = subprocess.run([str(binary), "--PrintGlobals"], check=True,
                                  capture_output=True, text=True, timeout=10).stdout
    if "--RngSeed=[1]" not in globals_help or "--RngRun=[1]" not in globals_help:
        raise RuntimeError("Compiled executable does not expose expected RNG globals")
    help_text = subprocess.run([str(binary), "--PrintHelp"], check=True,
                               capture_output=True, text=True, timeout=10).stdout
    required = ("--ueNumPergNb", "--bandwidthBand1", "--bandwidthBand2", "--totalTxPower",
                "--lambdaBe", "--lambdaUll", "--packetSizeBe", "--packetSizeUll",
                "--scenarioId", "--outputDir")
    absent = [arg for arg in required if arg not in help_text]
    if absent:
        raise RuntimeError(f"Compiled executable is missing required CLI arguments: {absent}")
    source = source_path.read_text(encoding="utf-8")
    for marker in ("SetUtNumber(ueNumPergNb * gNbNum)",
                   "Seconds(1.0 / lambdaBe)",
                   "udpPacketSizeBe",
                   "SetTxPower(10 * log10((bandwidthBand2 / totalBandwidth) * x))"):
        if marker not in source:
            raise RuntimeError(f"Scenario source no longer matches audited setting: {marker}")

    configs = expansion_configurations()
    output_root.mkdir(parents=True)
    binary_version = subprocess.run([str(binary), "--PrintVersion"], check=True,
                                    capture_output=True, text=True, timeout=10).stdout.strip()
    campaign = {
        "experiment": "ns3_qoe_expansion_20261026_r1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "11 configurations × 2 RNG repetitions; bounded ns-3 testbed expansion",
        "design_table": [dict(config_id=a, bandwidth_band1_mhz=b, total_tx_power_dbm=p, ue_count=u)
                         for a, b, p, u in EXPANSION_TABLE],
        "rng_seed": SEED,
        "rng_runs": list(RNG_RUNS),
        "rng_pairing_caveat": "same seed/run numbers do not guarantee identical realized random events across changed event sequences",
        "bandwidth_band2_mhz": 50,
        "traffic": {"lambda_be_pps": LAMBDA_BE, "packet_size_be_bytes": 1252,
                    "lambda_ull_pps": 10000, "packet_size_ull_bytes": 100,
                    "application_start_seconds": 0.4},
        "offered_load_definition": "lambda_pps * configured UDP payload bytes * 8 / 1e6; excludes protocol headers; no post-run TX counters",
        "offered_load_mbps_per_flow": {"low_latency": 8.0, "voice_be": 200.32},
        "fixed_scenario": {"gNb_count": 1, "scenario_size_m": 30, "sim_time": "1s",
                           "central_frequency_band1_hz": 28000000000,
                           "central_frequency_band2_hz": 28200000000,
                           "numerology_bwp1": 4, "numerology_bwp2": 2,
                           "double_operational_band": True},
        "power_allocation": "P_i[dBm]=P_total[dBm]+10*log10(BW_i/(BW1+BW2))",
        "timeout_seconds_per_run": timeout_seconds,
        "binary": str(binary), "binary_version": binary_version,
        "binary_sha256": pilot._source_hash(binary),
        "scenario_source": str(source_path),
        "scenario_source_sha256": pilot._source_hash(source_path),
        "python_version": sys.version, "platform": platform.platform(),
        "retry_policy": "no automatic retries; failures are preserved and reported",
        "configurations": configs,
    }
    pilot._write_json(output_root / "campaign_config.json", campaign)
    manifest_path = output_root / "run_manifest.csv"
    records: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    fatal_validation = False

    for config in configs:
        run_dir = output_root / "runs" / config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        command = pilot.build_command(binary, config, run_dir)
        p1, p2 = derive_effective_band_powers_dbm(
            config["total_tx_power_dbm"], config["bandwidth_band1_mhz"], config["bandwidth_band2_mhz"]
        )
        record: dict[str, Any] = {
            "config_id": config["config_id"], "repetition_id": config["repetition_id"],
            **{key: config[key] for key in pilot.MANIFEST_COLUMNS if key in config},
            "effective_band1_tx_power_dbm": p1, "effective_band2_tx_power_dbm": p2,
            "output_dir": str(run_dir), "csv_path": str(run_dir / "qoe_dataset.csv"),
            "summary_path": str(run_dir / "sim_summary.txt"), "command": command,
            "status": "RUNNING", "attempt_number": 1,
        }
        pilot._write_json(run_dir / "run_config.json", {
            "config_id": config["config_id"], "repetition_id": config["repetition_id"],
            "run": config, "expected_effective_band_powers_dbm": [p1, p2], "command": command,
        })
        record["start_utc"] = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=run_dir, capture_output=True, text=True,
                                       timeout=timeout_seconds, check=False)
            record["process_returncode"] = completed.returncode
            (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"ns-3 returned exit code {completed.returncode}")
            if not Path(record["csv_path"]).is_file() or not Path(record["summary_path"]).is_file():
                raise RuntimeError("Exit code 0 but expected CSV or summary is absent")
            metrics = pilot.validate_run_output(Path(record["csv_path"]), config)
            record["metrics"] = metrics
            record["status"] = "PASS"
            feature_rows.extend(_record_feature_rows(config, Path(record["csv_path"])))
        except subprocess.TimeoutExpired as exc:
            record["process_returncode"] = -2
            record["status"] = "TIMEOUT"
            record["error"] = f"Timed out after {timeout_seconds} seconds"
            for filename, content in (("stdout.log", exc.stdout), ("stderr.log", exc.stderr)):
                if isinstance(content, bytes):
                    content = content.decode(errors="replace")
                (run_dir / filename).write_text(content or "", encoding="utf-8")
        except Exception as exc:
            record.setdefault("process_returncode", -3)
            record["status"] = "FAIL"
            record["error"] = f"{type(exc).__name__}: {exc}"
            # An output-configuration mismatch means a shared setting is wrong;
            # stop rather than spend the remaining campaign on invalid runs.
            if "Applied setting mismatch" in str(exc) or "CSV schema mismatch" in str(exc):
                fatal_validation = True

        record["duration_seconds"] = time.monotonic() - started
        record["end_utc"] = datetime.now(timezone.utc).isoformat()
        pilot._write_json(run_dir / "run_config.json", {
            "config_id": config["config_id"], "repetition_id": config["repetition_id"],
            "run": config, "expected_effective_band_powers_dbm": [p1, p2],
            "command": command, "start_utc": record["start_utc"], "end_utc": record["end_utc"],
            "duration_seconds": record["duration_seconds"],
            "process_returncode": record["process_returncode"], "status": record["status"],
            "error": record.get("error"), "metrics": record.get("metrics"),
        })
        records.append(record)
        _write_manifest(manifest_path, records)
        print(f"{record['status']} {config['run_id']} ({record['duration_seconds']:.2f}s) "
              f"{record.get('error', '')}", flush=True)
        if fatal_validation:
            print("STOP: shared schema/configuration validation failure", flush=True)
            break

    validation = _validate_design_records(records, configs)
    validation.update({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": sum(float(r.get("duration_seconds", 0)) for r in records),
        "fatal_validation_stop": fatal_validation,
        "retry_policy": "no automatic retries",
        "data_quality_gate": validation["all_runs_pass"],
    })
    if validation["all_runs_pass"]:
        if len(feature_rows) != sum(int(c["ue_count"]) for c in configs):
            validation["data_quality_gate"] = False
            validation["all_runs_pass"] = False
            validation["feature_row_count_error"] = len(feature_rows)
        else:
            keys = [(r["config_id"], r["repetition_id"], int(r["flow_id"])) for r in feature_rows]
            validation["unique_configuration_repetition_flow_keys"] = len(keys) == len(set(keys))
            validation["flow_rows"] = len(feature_rows)
            if not validation["unique_configuration_repetition_flow_keys"]:
                validation["all_runs_pass"] = False
                validation["data_quality_gate"] = False
            else:
                columns = list(feature_rows[0])
                with (output_root / "expanded_flow_dataset.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(feature_rows)
    validation["status"] = "PASS" if validation["all_runs_pass"] and validation["data_quality_gate"] else "INCOMPLETE_OR_FAILED"
    pilot._write_json(output_root / "campaign_validation.json", validation)
    return 0 if validation["status"] == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--binary", type=Path, default=pilot.DEFAULT_BINARY)
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    raise SystemExit(run_expansion(args.output_root, args.binary, args.timeout_seconds))


if __name__ == "__main__":
    main()
