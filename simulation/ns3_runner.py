"""Subprocess-driven ns-3 5G NR simulation runner.

Executes ns-3 in an isolated run directory:
- Protects raw datasets and existing results from overwriting.
- Directly invokes the compiled binary with explicit CLI arguments.
- Captures stdout/stderr, execution time, and process exit status.
- Configurable timeout and robust error reporting.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default path to pre-built compiled ns-3 scratch binary
DEFAULT_NS3_BINARY = (
    Path(__file__).resolve().parent.parent
    / "ns-3.48"
    / "build"
    / "scratch"
    / "ns3.48-qoe-dataset-default"
)


@dataclass
class Ns3RunResult:
    run_id: str
    success: bool
    exit_code: int
    run_dir: Path
    csv_path: Path
    summary_path: Path
    stdout: str
    stderr: str
    duration_seconds: float
    command: list[str]
    config: dict[str, Any]
    error_message: str | None = None


class Ns3Runner:
    """Runner that launches individual ns-3 5G simulations in isolated directories."""

    def __init__(
        self,
        executable_path: str | Path | None = None,
        default_output_base: str | Path = "results/phase3/runs",
        default_timeout_seconds: float = 120.0,
    ) -> None:
        self.executable_path = (
            Path(executable_path) if executable_path else DEFAULT_NS3_BINARY
        )
        self.output_base = Path(default_output_base).resolve()
        self.default_timeout = default_timeout_seconds

    def check_executable_available(self) -> tuple[bool, str]:
        """Check if the ns-3 compiled binary exists and is executable."""
        if not self.executable_path.exists():
            return False, f"ns-3 binary not found at {self.executable_path}"
        if not os.access(self.executable_path, os.X_OK):
            return False, f"ns-3 binary at {self.executable_path} is not executable"
        return True, "Executable available"

    def build_cli_arguments(
        self,
        config: dict[str, Any],
        run_dir: Path,
        sim_tag: str = "sim_summary.txt",
    ) -> list[str]:
        """Construct the exact CLI argument list supported by scratch/qoe-dataset.cc.
        
        Maps configuration keys to ns-3 CommandLine parameters:
            - ue_count -> --ueNumPergNb
            - bandwidth_mhz -> --bandwidthBand1 (in Hz)
            - total_tx_power_dbm -> --totalTxPower
            - scenario_size_m -> --scenarioSize
            - lambda_be -> --lambdaBe
            - scenario_id -> --scenarioId
            - sim_time_str -> --simTime
            - run_dir -> --outputDir
            - sim_tag -> --simTag
        """
        ue_count = int(config.get("ue_count", 2))
        bw_hz = int(float(config.get("bandwidth_mhz", 50.0)) * 1_000_000)
        power_dbm = float(config.get("total_tx_power_dbm", 35.0))
        size_m = float(config.get("scenario_size_m", 30.0))
        lambda_be = int(float(config.get("lambda_be", 10000.0)))
        scenario_id = int(config.get("scenario_id", 1))
        sim_time = str(config.get("sim_time", "1s"))

        args = [
            str(self.executable_path),
            f"--ueNumPergNb={ue_count}",
            f"--bandwidthBand1={bw_hz}",
            f"--totalTxPower={power_dbm}",
            f"--scenarioSize={size_m}",
            f"--lambdaBe={lambda_be}",
            f"--scenarioId={scenario_id}",
            f"--simTime={sim_time}",
            f"--outputDir={str(run_dir)}",
            f"--simTag={sim_tag}",
        ]
        return args

    def run_simulation(
        self,
        config: dict[str, Any],
        run_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Ns3RunResult:
        """Run a single ns-3 simulation in a dedicated run directory.
        
        Parameters:
            config: Network configuration dictionary.
            run_id: Optional unique identifier. Defaults to timestamp-based ID.
            timeout_seconds: Max seconds to wait for simulation completion.
        """
        if run_id is None:
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:19]
            run_id = f"run_{now_str}"

        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout

        run_dir = self.output_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        sim_tag = "sim_summary.txt"
        cmd = self.build_cli_arguments(config, run_dir, sim_tag=sim_tag)

        # Save run configuration metadata
        config_path = run_dir / "run_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config": config,
                    "command": cmd,
                },
                f,
                indent=2,
            )

        start_time = time.monotonic()
        stdout_text = ""
        stderr_text = ""
        exit_code = -1
        error_msg = None

        try:
            # Run simulation with cwd=run_dir so that qoe_dataset.csv is created in run_dir
            proc = subprocess.run(
                cmd,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout_text = proc.stdout
            stderr_text = proc.stderr
            exit_code = proc.returncode

            if exit_code != 0:
                error_msg = f"ns-3 process exited with non-zero code {exit_code}: {stderr_text[:300]}"

        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start_time
            error_msg = f"ns-3 simulation timed out after {timeout} seconds"
            stdout_text = exc.stdout or "" if isinstance(exc.stdout, str) else ""
            stderr_text = exc.stderr or "" if isinstance(exc.stderr, str) else ""
            return Ns3RunResult(
                run_id=run_id,
                success=False,
                exit_code=-2,
                run_dir=run_dir,
                csv_path=run_dir / "qoe_dataset.csv",
                summary_path=run_dir / sim_tag,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=duration,
                command=cmd,
                config=config,
                error_message=error_msg,
            )
        except Exception as exc:
            duration = time.monotonic() - start_time
            error_msg = f"Unexpected runner error: {exc}"
            return Ns3RunResult(
                run_id=run_id,
                success=False,
                exit_code=-3,
                run_dir=run_dir,
                csv_path=run_dir / "qoe_dataset.csv",
                summary_path=run_dir / sim_tag,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=duration,
                command=cmd,
                config=config,
                error_message=error_msg,
            )

        duration = time.monotonic() - start_time

        # Save stdout and stderr logs in run directory
        with open(run_dir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(run_dir / "stderr.log", "w", encoding="utf-8") as f:
            f.write(stderr_text)

        csv_path = run_dir / "qoe_dataset.csv"
        summary_path = run_dir / sim_tag

        success = (exit_code == 0) and csv_path.exists()
        if not success and error_msg is None:
            if not csv_path.exists():
                error_msg = f"ns-3 succeeded with exit code 0 but {csv_path} was not created"

        return Ns3RunResult(
            run_id=run_id,
            success=success,
            exit_code=exit_code,
            run_dir=run_dir,
            csv_path=csv_path,
            summary_path=summary_path,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            command=cmd,
            config=config,
            error_message=error_msg,
        )
