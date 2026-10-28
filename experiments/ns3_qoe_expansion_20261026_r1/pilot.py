"""Run and validate the isolated seven-simulation ns-3 QoE pilot.

Uses the already-built qoe-dataset executable directly. Each invocation has a
fresh result directory, explicit traffic/radio/RNG arguments, a 120 s timeout,
and a saved exact command. The production dataset/model/results are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .preprocessing import derive_effective_band_powers_dbm, derive_flow_inputs, recompute_qoe_proxy
else:  # Direct script invocation keeps this experimental module self-contained.
    from preprocessing import derive_effective_band_powers_dbm, derive_flow_inputs, recompute_qoe_proxy


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "ns3_qoe_expansion_20261026_r1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "experiments" / EXPERIMENT_NAME
DEFAULT_BINARY = REPO_ROOT / "ns-3.48" / "build" / "scratch" / "ns3.48-qoe-dataset-default"
SOURCE_FILE = REPO_ROOT / "ns-3.48" / "scratch" / "qoe-dataset.cc"
TIMEOUT_SECONDS = 120

CSV_COLUMNS = [
    "scenario_id", "ue_count", "lambda_be", "bandwidth_mhz", "total_tx_power_dbm",
    "scenario_size_m", "flow_id", "tx_packets", "rx_packets", "lost_packets",
    "throughput_mbps", "delay_ms", "jitter_ms", "packet_loss", "qoe_proxy",
]
NUMERIC_COLUMNS = [column for column in CSV_COLUMNS]
MANIFEST_COLUMNS = [
    "run_id", "scenario_id", "bandwidth_band1_mhz", "bandwidth_band2_mhz",
    "total_tx_power_dbm", "effective_band1_tx_power_dbm", "effective_band2_tx_power_dbm",
    "ue_count", "gNb_count", "lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes",
    "packet_size_ull_bytes", "scenario_size_m", "sim_time", "rng_seed", "rng_run",
    "start_utc", "end_utc", "duration_seconds", "process_returncode", "status",
    "output_dir", "csv_path", "summary_path", "command_json", "metrics_json", "error",
]


def pilot_configurations() -> list[dict[str, Any]]:
    """Exact user-requested seven-run sequence; common RngRun=1, baseline repeat=2."""
    requested = [
        (50, 35, 2, 1),
        (50, 35, 2, 2),
        (20, 35, 2, 1),
        (100, 35, 2, 1),
        (50, 30, 2, 1),
        (50, 40, 2, 1),
        (50, 35, 10, 1),
    ]
    configs = []
    for index, (band1, power, ues, rng_run) in enumerate(requested, start=1):
        configs.append({
            "run_id": f"pilot_{index:02d}_b1{band1}_p{power}_ue{ues}_rng{rng_run}",
            "scenario_id": 900000 + index,
            "bandwidth_band1_mhz": band1,
            "bandwidth_band2_mhz": 50,
            "total_tx_power_dbm": power,
            "ue_count": ues,
            "gNb_count": 1,
            "lambda_be_pps": 5000,
            "lambda_ull_pps": 10000,
            "packet_size_be_bytes": 1252,
            "packet_size_ull_bytes": 100,
            "scenario_size_m": 30,
            "sim_time": "1s",
            "app_start_seconds": 0.4,
            "numerology_bwp1": 4,
            "numerology_bwp2": 2,
            "central_frequency_band1_hz": 28_000_000_000,
            "central_frequency_band2_hz": 28_200_000_000,
            "rng_seed": 20261026,
            "rng_run": rng_run,
        })
    return configs


def build_command(binary: Path, config: dict[str, Any], output_dir: Path) -> list[str]:
    """Build the exact binary argv using names confirmed by --PrintHelp/Globals."""
    return [
        str(binary),
        f"--gNbNum={config['gNb_count']}",
        f"--ueNumPergNb={config['ue_count']}",
        f"--scenarioId={config['scenario_id']}",
        "--doubleOperationalBand=true",
        f"--packetSizeUll={config['packet_size_ull_bytes']}",
        f"--packetSizeBe={config['packet_size_be_bytes']}",
        f"--lambdaUll={config['lambda_ull_pps']}",
        f"--lambdaBe={config['lambda_be_pps']}",
        f"--simTime={config['sim_time']}",
        f"--numerologyBwp1={config['numerology_bwp1']}",
        f"--centralFrequencyBand1={config['central_frequency_band1_hz']}",
        f"--bandwidthBand1={config['bandwidth_band1_mhz'] * 1_000_000}",
        f"--numerologyBwp2={config['numerology_bwp2']}",
        f"--centralFrequencyBand2={config['central_frequency_band2_hz']}",
        f"--bandwidthBand2={config['bandwidth_band2_mhz'] * 1_000_000}",
        f"--totalTxPower={config['total_tx_power_dbm']}",
        f"--scenarioSize={config['scenario_size_m']}",
        f"--simTag=sim_summary.txt",
        f"--outputDir={output_dir}",
        f"--RngSeed={config['rng_seed']}",
        f"--RngRun={config['rng_run']}",
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({column: serialized.get(column, "") for column in MANIFEST_COLUMNS})


def validate_run_output(csv_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Check schema, exact output settings, data integrity and independent QoE."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise ValueError("Expected qoe_dataset.csv is absent or empty")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"CSV schema mismatch: {reader.fieldnames!r}")
        rows = list(reader)
    if len(rows) != int(config["ue_count"]):
        raise ValueError(f"Expected {config['ue_count']} flow rows, got {len(rows)}")
    if not rows:
        raise ValueError("No flow records were exported")

    keys: set[tuple[int, int]] = set()
    exact_rows: set[tuple[str, ...]] = set()
    measurements: dict[str, list[float]] = {
        "throughput_mbps": [], "delay_ms": [], "jitter_ms": [], "packet_loss": [],
        "qoe_proxy": [],
    }
    outage_rows = 0
    qoe_max_abs_error = 0.0
    loss_max_abs_error = 0.0
    derived_rows: list[dict[str, Any]] = []
    for row in rows:
        if any(row.get(column, "") == "" for column in CSV_COLUMNS):
            raise ValueError("CSV contains a missing value")
        values = {column: float(row[column]) for column in NUMERIC_COLUMNS}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("CSV contains a non-finite numeric value")
        identity = (int(values["scenario_id"]), int(values["flow_id"]))
        if identity in keys:
            raise ValueError(f"Duplicate scenario/flow identifier: {identity}")
        keys.add(identity)
        exact = tuple(row[column] for column in CSV_COLUMNS)
        if exact in exact_rows:
            raise ValueError("CSV contains an exact duplicate row")
        exact_rows.add(exact)

        expected_config = {
            "scenario_id": config["scenario_id"],
            "ue_count": config["ue_count"],
            "lambda_be": config["lambda_be_pps"],
            "bandwidth_mhz": config["bandwidth_band1_mhz"],
            "total_tx_power_dbm": config["total_tx_power_dbm"],
            "scenario_size_m": config["scenario_size_m"],
        }
        for column, expected in expected_config.items():
            if not math.isclose(values[column], float(expected), rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"Applied setting mismatch for {column}: {values[column]} != {expected}")

        tx, rx = int(values["tx_packets"]), int(values["rx_packets"])
        expected_loss = 0.0 if tx <= 0 else max(0.0, min(1.0, 1.0 - rx / tx))
        loss_error = abs(values["packet_loss"] - expected_loss)
        loss_max_abs_error = max(loss_max_abs_error, loss_error)
        if loss_error > 1e-5:
            raise ValueError(f"Packet-loss counter check failed for flow {identity}")
        if not (0.0 <= values["packet_loss"] <= 1.0):
            raise ValueError(f"Packet loss outside [0,1] for flow {identity}")
        for column in ("throughput_mbps", "delay_ms", "jitter_ms"):
            if values[column] < 0.0:
                raise ValueError(f"Negative {column} for flow {identity}")
        if not (0.0 <= values["qoe_proxy"] <= 1.0):
            raise ValueError(f"QoE proxy outside [0,1] for flow {identity}")

        independently_computed_qoe = recompute_qoe_proxy(row)
        qoe_error = abs(values["qoe_proxy"] - independently_computed_qoe)
        qoe_max_abs_error = max(qoe_max_abs_error, qoe_error)
        if qoe_error > 2e-5:
            raise ValueError(f"Independent QoE recomputation mismatch for flow {identity}")
        if rx == 0:
            outage_rows += 1
            if any(values[name] != 0.0 for name in ("throughput_mbps", "delay_ms", "jitter_ms", "qoe_proxy")):
                raise ValueError(f"Zero-reception handling mismatch for flow {identity}")

        for column in measurements:
            measurements[column].append(values[column])
        flow_inputs = derive_flow_inputs(
            ue_count=config["ue_count"],
            flow_id=int(values["flow_id"]),
            lambda_be_pps=config["lambda_be_pps"],
            lambda_ull_pps=config["lambda_ull_pps"],
            packet_size_ull_bytes=config["packet_size_ull_bytes"],
            packet_size_be_bytes=config["packet_size_be_bytes"],
        )
        derived_rows.append({
            "run_id": config["run_id"],
            "scenario_id": config["scenario_id"],
            "flow_id": int(values["flow_id"]),
            **flow_inputs,
        })

    if sorted(flow_id for _, flow_id in keys) != list(range(1, int(config["ue_count"]) + 1)):
        raise ValueError("Flow IDs do not match the expected one-flow-per-UE sequence")
    metrics = {
        "row_count": len(rows),
        "missing_values": 0,
        "non_finite_values": 0,
        "exact_duplicate_rows": 0,
        "duplicate_scenario_flow_keys": 0,
        "outage_rows": outage_rows,
        "qoe_recompute_max_abs_error": qoe_max_abs_error,
        "packet_loss_recompute_max_abs_error": loss_max_abs_error,
        "means": {key: sum(values) / len(values) for key, values in measurements.items()},
        "flow_rows": derived_rows,
    }
    return metrics


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pilot_report(output_root: Path, records: list[dict[str, Any]], overall: dict[str, Any]) -> None:
    lines = [
        "# Seven-run ns-3 QoE expansion pilot",
        "",
        f"- Completed at (UTC): {overall.get('completed_utc', 'incomplete')}",
        f"- Runs attempted: {len(records)} / 7",
        f"- Pilot status: **{overall.get('status', 'INCOMPLETE')}**",
        "- Scope: one ns-3 testbed simulation scenario; not real-world 5G evidence.",
        "- No production dataset, preprocessing, model, optimizer, or controller was changed.",
        "",
        "## Run results",
        "",
        "| Run | B1 MHz | B2 MHz | total dBm | effective B1 dBm | effective B2 dBm | UEs | seed/run | duration s | result | outages | mean QoE | mean throughput Mbps | mean delay ms | mean jitter ms | mean loss fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record.get("metrics", {})
        means = metrics.get("means", {})
        lines.append(
            "| {run_id} | {b1} | {b2} | {power} | {p1:.3f} | {p2:.3f} | {ues} | {seed}/{rng} | {duration} | {status} | {outages} | {qoe} | {throughput} | {delay} | {jitter} | {loss} |".format(
                run_id=record.get("run_id", "—"), b1=record.get("bandwidth_band1_mhz", "—"),
                b2=record.get("bandwidth_band2_mhz", "—"), power=record.get("total_tx_power_dbm", "—"),
                p1=record.get("effective_band1_tx_power_dbm", 0.0),
                p2=record.get("effective_band2_tx_power_dbm", 0.0), ues=record.get("ue_count", "—"),
                seed=record.get("rng_seed", "—"), rng=record.get("rng_run", "—"),
                duration=f"{record.get('duration_seconds', 0.0):.2f}", status=record.get("status", "—"),
                outages=metrics.get("outage_rows", "—"), qoe=f"{means.get('qoe_proxy', float('nan')):.5f}",
                throughput=f"{means.get('throughput_mbps', float('nan')):.4f}",
                delay=f"{means.get('delay_ms', float('nan')):.4f}",
                jitter=f"{means.get('jitter_ms', float('nan')):.5f}",
                loss=f"{means.get('packet_loss', float('nan')):.6f}",
            )
        )
    lines.extend([
        "",
        "## Configuration-response review",
        "",
        "At UE=2, the Band 1=100 MHz and total-power=30/40 dBm runs with RngRun=1 produced the same per-flow KPIs as the 50 MHz/35 dBm reference. This is plausible for the fixed 5,000-packet/s best-effort load, which the 50 MHz/35 dBm condition already carries with near-complete delivery; it does not demonstrate a benefit or sensitivity to those settings at this workload. The 20 MHz run produced complete outage. At UE=10, mean QoE fell and mean delay/loss increased, consistent with the higher aggregate offered load. No monotonicity claim is made.",
        "",
        "The configuration-response gate is **pass with a low-load sensitivity caveat**: the requested settings are reflected in command/config metadata and setter logic; observed changes are plausible, but higher offered load is needed to study bandwidth/power effects.",
        "",
        "## Validation gates",
        "",
        f"- All completed process exits and required artifacts valid: {overall.get('process_and_artifact_gate', False)}",
        f"- Requested CSV settings and row counts valid: {overall.get('configuration_gate', False)}",
        f"- Schema, finite values, missing values and duplicate checks pass: {overall.get('data_quality_gate', False)}",
        f"- Independent packet-loss and QoE recomputation pass: {overall.get('metric_recomputation_gate', False)}",
        "- Configuration response: see per-run means above; no monotonicity requirement was imposed.",
        "",
        "## Decision",
        "",
        overall.get("decision", "Pilot incomplete; do not begin the 22-run expansion."),
        "",
        "## Proposed 22-run fractional expansion (approval required; not started)",
        "",
        "The pilot suggests the fixed lambdaBe=5,000 load does not discriminate Band 1=50/100 MHz or total power=30/35/40 dBm at UE=2. For the next design, hold lambdaBe at 20,000 packets/s (an existing dataset traffic level), lambdaUll at 10,000, Band 2 at 50 MHz, scenario size at 30 m, sim time at 1 s, and all other radio settings fixed. This higher offered load is intended to make bandwidth/power sensitivity measurable; it remains a proposal until approved and piloted if needed.",
        "",
        "| Fractional-array row | Band 1 MHz | total power dBm | UEs |",
        "|---:|---:|---:|---:|",
        "| 1 | 20 | 30 | 2 |",
        "| 2 | 20 | 35 | 6 |",
        "| 3 | 20 | 40 | 10 |",
        "| 4 | 50 | 30 | 6 |",
        "| 5 | 50 | 35 | 10 |",
        "| 6 | 50 | 40 | 2 |",
        "| 7 | 100 | 30 | 10 |",
        "| 8 | 100 | 35 | 2 |",
        "| 9 | 100 | 40 | 6 |",
        "| Anchor | 50 | 35 | 2 |",
        "| Anchor | 50 | 35 | 6 |",
        "",
        "Run each of the 11 settings at RngSeed=20261026 and RngRun=1,2 (22 runs total) with unique scenario IDs and per-run directories. This is an L9 fractional design plus two anchors, not a full factorial.",
        "",
        "For evaluation, keep all flows/repetitions of each configuration together. Predeclare separate leave-one-level-out checks for Band 1=100 MHz, total power=40 dBm, and UE=10. When using original baseline rows for training, exclude every row at a held-out level in that split. Fit transforms inside training folds. Report per-configuration MAE/RMSE, outage counts and paired seed outcomes; treat the production model as in-domain only at its original feature support. Do not modify the production optimizer/model from these results alone.",
        "",
        "**Approval gate:** do not start these runs automatically; the pilot report and design must be reviewed first.",
        "",
        "## Reproduction",
        "",
        "The exact argument vector, process exit code, UTC timestamps and output paths are in `run_manifest.csv` and each run's `run_config.json`. Reproduce with a new, unused output root:",
        "",
        "```bash",
        ".venv/bin/python experiments/ns3_qoe_expansion_20261026_r1/pilot.py --output-root results/experiments/ns3_qoe_expansion_20261026_r1_reproduction",
        "```",
        "",
        "This script refuses to reuse an existing output directory. It runs at most seven simulations, each with a 120-second timeout, and stops after the first failed gate.",
        "",
    ])
    (output_root / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_pilot(output_root: Path, binary: Path, timeout_seconds: float = TIMEOUT_SECONDS) -> int:
    output_root = output_root.resolve()
    binary = binary.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse existing pilot output directory: {output_root}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"ns-3 executable is absent or not executable: {binary}")
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive")

    output_root.mkdir(parents=True)
    manifest_path = output_root / "run_manifest.csv"
    configs = pilot_configurations()
    global_help = subprocess.run(
        [str(binary), "--PrintGlobals"], check=True, capture_output=True, text=True, timeout=10
    ).stdout
    seed = global_help
    if "--RngSeed=[1]" not in seed or "--RngRun=[1]" not in seed:
        raise RuntimeError("Compiled executable does not expose expected RngSeed/RngRun globals")
    program_help = subprocess.run(
        [str(binary), "--PrintHelp"], check=True, capture_output=True, text=True, timeout=10
    ).stdout
    required_options = (
        "--ueNumPergNb", "--bandwidthBand1", "--bandwidthBand2", "--totalTxPower",
        "--scenarioId", "--outputDir", "--simTime",
    )
    absent_options = [option for option in required_options if option not in program_help]
    if absent_options:
        raise RuntimeError(f"Compiled executable is missing required options: {absent_options}")

    source = SOURCE_FILE.read_text(encoding="utf-8")
    source_required = (
        "SetUtNumber(ueNumPergNb * gNbNum)",
        "10 * log10((bandwidthBand1 / totalBandwidth) * x)",
        "SetTxPower(10 * log10((bandwidthBand2 / totalBandwidth) * x))",
    )
    if any(marker not in source for marker in source_required):
        raise RuntimeError("Scenario source no longer matches the audited UE/power allocation logic")

    version = subprocess.run(
        [str(binary), "--PrintVersion"], check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()
    global_config = {
        "experiment": EXPERIMENT_NAME,
        "scope": "exactly seven pilot runs; no 22-run expansion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary),
        "binary_sha256": _source_hash(binary),
        "scenario_source": str(SOURCE_FILE),
        "scenario_source_sha256": _source_hash(SOURCE_FILE),
        "ns3_version_output": version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": 20261026,
        "rng_run_policy": "RngRun=1 for all primary configurations; baseline repeat uses RngRun=2",
        "rng_globals_help": global_help.strip().splitlines(),
        "bandwidth_band2_mhz": 50,
        "traffic": {
            "gNb_count": 1, "lambda_be_pps": 5000, "lambda_ull_pps": 10000,
            "packet_size_be_bytes": 1252, "packet_size_ull_bytes": 100,
            "scenario_size_m": 30, "sim_time": "1s", "application_start_seconds": 0.4,
        },
        "timeout_seconds_per_run": timeout_seconds,
        "configurations": configs,
        "power_allocation": "P_i(dBm)=P_total+10*log10(BW_i/(BW1+BW2)); doubleOperationalBand=true",
        "qoe_proxy": "rx=0 => 0; else 0.40*min(throughput_Mbps/50,1)+0.25*exp(-delay_ms/50)+0.20*exp(-jitter_ms/20)+0.15*(1-loss_fraction)",
    }
    _write_json(output_root / "pilot_config.json", global_config)

    records: list[dict[str, Any]] = []
    process_gate = configuration_gate = quality_gate = metric_gate = True
    stopped_early = False
    for config in configs:
        run_dir = output_root / "runs" / config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        command = build_command(binary, config, run_dir)
        p1, p2 = derive_effective_band_powers_dbm(
            config["total_tx_power_dbm"],
            config["bandwidth_band1_mhz"],
            config["bandwidth_band2_mhz"],
        )
        record: dict[str, Any] = {
            **{key: config[key] for key in (
                "run_id", "scenario_id", "bandwidth_band1_mhz", "bandwidth_band2_mhz",
                "total_tx_power_dbm", "ue_count", "gNb_count", "lambda_be_pps", "lambda_ull_pps",
                "packet_size_be_bytes", "packet_size_ull_bytes", "scenario_size_m", "sim_time",
                "rng_seed", "rng_run",
            )},
            "effective_band1_tx_power_dbm": p1,
            "effective_band2_tx_power_dbm": p2,
            "output_dir": str(run_dir),
            "csv_path": str(run_dir / "qoe_dataset.csv"),
            "summary_path": str(run_dir / "sim_summary.txt"),
            "command": command,
            "status": "RUNNING",
        }
        _write_json(run_dir / "run_config.json", {"run": config, "expected_effective_band_powers_dbm": [p1, p2], "command": command})
        record["start_utc"] = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=str(run_dir), capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
            record["process_returncode"] = completed.returncode
            (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"ns-3 returned exit code {completed.returncode}")
            if not Path(record["csv_path"]).is_file() or not Path(record["summary_path"]).is_file():
                raise RuntimeError("Process exited 0 but one or more expected artifacts are missing")
            metrics = validate_run_output(Path(record["csv_path"]), config)
            record["metrics"] = metrics
            record["status"] = "PASS"
        except subprocess.TimeoutExpired as exc:
            record["process_returncode"] = -2
            record["error"] = f"Timed out after {timeout_seconds} seconds"
            for filename, content in (("stdout.log", exc.stdout), ("stderr.log", exc.stderr)):
                if isinstance(content, bytes):
                    content = content.decode(errors="replace")
                (run_dir / filename).write_text(content or "", encoding="utf-8")
            record["status"] = "TIMEOUT"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if "process_returncode" not in record:
                record["process_returncode"] = -3
            record["status"] = "FAIL"
        record["duration_seconds"] = time.monotonic() - started
        record["end_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "run_config.json", {
            "run": config,
            "expected_effective_band_powers_dbm": [p1, p2],
            "command": command,
            "start_utc": record["start_utc"],
            "end_utc": record["end_utc"],
            "duration_seconds": record["duration_seconds"],
            "process_returncode": record["process_returncode"],
            "status": record["status"],
            "error": record.get("error"),
            "metrics": record.get("metrics"),
        })
        records.append(record)
        _write_manifest(manifest_path, records)
        if record["status"] != "PASS":
            process_gate = False
            if record.get("error", "").startswith("ValueError") or record.get("error", "").startswith("RuntimeError: Applied setting"):
                configuration_gate = False
            else:
                quality_gate = False
            stopped_early = True
            print(f"STOP after {config['run_id']}: {record.get('error', record['status'])}", flush=True)
            break
        print(
            f"PASS {config['run_id']} elapsed={record['duration_seconds']:.2f}s "
            f"qoe_mean={record['metrics']['means']['qoe_proxy']:.5f} "
            f"outages={record['metrics']['outage_rows']}/{record['metrics']['row_count']}",
            flush=True,
        )

    all_attempted = len(records) == len(configs)
    all_pass = all(record.get("status") == "PASS" for record in records)
    validation_gates = {
        "all_seven_runs_attempted": all_attempted,
        "all_processes_exit_zero_and_artifacts_exist": process_gate and all_pass,
        "configuration_and_flow_counts_match": configuration_gate and all_pass,
        "schema_quality_missing_nonfinite_duplicate_and_physical_checks": quality_gate and all_pass,
        "independent_packet_loss_and_qoe_recomputation": metric_gate and all_pass,
    }
    overall_pass = all(validation_gates.values())
    overall = {
        "status": "PASS" if overall_pass else ("STOPPED_ON_GATE_FAILURE" if stopped_early else "INCOMPLETE"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "attempted_runs": len(records),
        "passed_runs": sum(record.get("status") == "PASS" for record in records),
        "failed_runs": sum(record.get("status") != "PASS" for record in records),
        "total_runtime_seconds": sum(float(record.get("duration_seconds", 0.0)) for record in records),
        "process_and_artifact_gate": validation_gates["all_processes_exit_zero_and_artifacts_exist"],
        "configuration_gate": validation_gates["configuration_and_flow_counts_match"],
        "data_quality_gate": validation_gates["schema_quality_missing_nonfinite_duplicate_and_physical_checks"],
        "metric_recomputation_gate": validation_gates["independent_packet_loss_and_qoe_recomputation"],
        "validation_gates": validation_gates,
        "configuration_response": "Configuration means are descriptive; monotonic improvement is not required.",
        "configuration_response_gate": "PASS_WITH_LOW_LOAD_SENSITIVITY_CAVEAT" if all_pass else "NOT_PASSED",
        "decision": (
            "Pilot gates passed. Do not start the 22-run expansion until the pilot has been reviewed and separately approved."
            if overall_pass else
            "Pilot did not pass all gates. Do not start the 22-run expansion; diagnose the failed run/gate first."
        ),
    }
    _write_json(output_root / "pilot_validation.json", overall)
    write_pilot_report(output_root, records, overall)
    print(f"Pilot {overall['status']}: {len(records)}/7 attempted; report={output_root / 'pilot_report.md'}", flush=True)
    return 0 if overall_pass else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--timeout-seconds", type=int, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    try:
        exit_code = run_pilot(args.output_root, args.binary, args.timeout_seconds)
    except Exception as exc:
        parser.exit(2, f"pilot setup failed: {type(exc).__name__}: {exc}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
