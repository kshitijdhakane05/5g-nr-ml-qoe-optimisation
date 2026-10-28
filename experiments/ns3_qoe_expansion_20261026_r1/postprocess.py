"""Export validated pilot rows with only configuration-derived load features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .pilot import CSV_COLUMNS, DEFAULT_OUTPUT_ROOT, validate_run_output
from .preprocessing import derive_effective_band_powers_dbm, derive_flow_inputs


EXTRA_COLUMNS = [
    "run_id", "configuration_id", "rng_seed", "rng_run", "bandwidth_band1_mhz",
    "bandwidth_band2_mhz", "effective_band1_tx_power_dbm", "effective_band2_tx_power_dbm",
    "lambda_ull_pps", "flow_type", "flow_type_id", "flow_lambda", "packet_size_bytes",
    "offered_throughput_mbps",
]


def export_pilot_table(pilot_root: Path, output_path: Path | None = None) -> Path:
    pilot_root = pilot_root.resolve()
    validation_path = pilot_root / "pilot_validation.json"
    config_path = pilot_root / "pilot_config.json"
    manifest_path = pilot_root / "run_manifest.csv"
    for required in (validation_path, config_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required pilot artifact not found: {required}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("Refusing to export training-style rows from a failed/incomplete pilot")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    configs = {config["run_id"]: config for config in metadata["configurations"]}
    output_path = (output_path or (pilot_root / "pilot_preprocessed_configured_load.csv")).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing feature table: {output_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 7 or any(record["status"] != "PASS" for record in manifest):
        raise ValueError("Expected exactly seven validated runs in the pilot manifest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = EXTRA_COLUMNS + CSV_COLUMNS
    with output_path.open("x", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest:
            run_id = record["run_id"]
            config = configs.get(run_id)
            if config is None:
                raise ValueError(f"Run {run_id} is missing from pilot_config.json")
            run_csv = Path(record["csv_path"])
            validate_run_output(run_csv, config)
            p1, p2 = derive_effective_band_powers_dbm(
                config["total_tx_power_dbm"], config["bandwidth_band1_mhz"],
                config["bandwidth_band2_mhz"],
            )
            with run_csv.open("r", encoding="utf-8-sig", newline="") as in_handle:
                for row in csv.DictReader(in_handle):
                    flow = derive_flow_inputs(
                        ue_count=config["ue_count"], flow_id=int(row["flow_id"]),
                        lambda_be_pps=config["lambda_be_pps"],
                        lambda_ull_pps=config["lambda_ull_pps"],
                        packet_size_ull_bytes=config["packet_size_ull_bytes"],
                        packet_size_be_bytes=config["packet_size_be_bytes"],
                    )
                    flow_id = int(row["flow_id"])
                    lowlat = flow["flow_type"] == "low_latency"
                    configuration_id = (
                        f"ue{config['ue_count']}_b1{config['bandwidth_band1_mhz']}"
                        f"_b2{config['bandwidth_band2_mhz']}_p{config['total_tx_power_dbm']}"
                        f"_lbe{config['lambda_be_pps']}_size{config['scenario_size_m']}"
                    )
                    writer.writerow({
                        "run_id": run_id,
                        "configuration_id": configuration_id,
                        "rng_seed": config["rng_seed"],
                        "rng_run": config["rng_run"],
                        "bandwidth_band1_mhz": config["bandwidth_band1_mhz"],
                        "bandwidth_band2_mhz": config["bandwidth_band2_mhz"],
                        "effective_band1_tx_power_dbm": f"{p1:.9f}",
                        "effective_band2_tx_power_dbm": f"{p2:.9f}",
                        "lambda_ull_pps": config["lambda_ull_pps"],
                        "flow_type": flow["flow_type"],
                        "flow_type_id": 0 if lowlat else 1,
                        "flow_lambda": flow["flow_lambda_pps"],
                        "packet_size_bytes": flow["packet_size_bytes"],
                        "offered_throughput_mbps": f"{flow['offered_throughput_mbps_configured']:.9f}",
                        **row,
                    })
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(export_pilot_table(args.pilot_root, args.output))


if __name__ == "__main__":
    main()
