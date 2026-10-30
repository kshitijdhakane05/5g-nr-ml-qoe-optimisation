"""Display the external campus dataset benchmark without conflating domains."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models/campus_5g_zenodo_13754300"
PROVENANCE = ROOT / "data/raw/5g_campus_network_qos_zenodo_13754300/PROVENANCE.json"


def render_external_dataset_page(st) -> None:
    st.title("External 5G Campus Dataset Benchmark")
    st.caption("Research benchmark only · predictions are not ns-3 measurements")
    if not PROVENANCE.exists() or not (ARTIFACT_DIR / "evaluation.csv").exists():
        st.info("The external dataset benchmark has not been trained yet.")
        return

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    metrics = pd.read_csv(ARTIFACT_DIR / "evaluation.csv")
    audit = json.loads((ARTIFACT_DIR / "audit.json").read_text(encoding="utf-8"))
    st.markdown(
        f"Source: [{provenance['title']}]({provenance['source_url']}) · "
        f"retrieved {provenance['retrieved']} · {provenance['license']} · "
        f"DOI [{provenance['doi']}](https://doi.org/{provenance['doi']})"
    )
    cols = st.columns(3)
    cols[0].metric("Training runs", f"{audit['rows']:,}")
    cols[1].metric("Configuration groups", str(audit["configuration_groups"]))
    cols[2].metric("Evaluation", f"{audit['folds']}-fold grouped CV" if "folds" in audit else "5-fold grouped CV")
    st.warning(
        "This model covers NTNU/Würzburg campus OAI/srsRAN tests only. Its bandwidth, slot allocation, "
        "DL:UL ratio, gNB, and SDR values have no verified mapping to the project's ns-3 controls. "
        "Do not use these predictions to rank or label ns-3 configurations."
    )
    st.subheader("Grouped out-of-fold benchmark scores")
    st.caption("MAE, RMSE, and R² are computed on held-out configuration groups. Values remain in source-reported units.")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.subheader("Prediction targets and limits")
    st.write(
        "Separate targets: actual uplink/downlink throughput, mean uplink/downlink jitter, and mean uplink/downlink loss. "
        "The source conflicts between MBps labels and `mbps` field names for throughput, and does not define the loss scale. "
        "No human QoE label is supplied; no QoE score is inferred here."
    )
    for limitation in audit["limitations"]:
        st.markdown(f"- {limitation}")
    if (ARTIFACT_DIR / "grouped_cv_predictions.csv").exists():
        with st.expander("Grouped out-of-fold predictions (model estimates, not observations)"):
            predictions = pd.read_csv(ARTIFACT_DIR / "grouped_cv_predictions.csv")
            st.dataframe(predictions, use_container_width=True, hide_index=True)


def render_live_model_provenance(st) -> None:
    """Show exactly which data source informs the live controller's model advice."""
    st.title("Live Hybrid Model & Dataset Provenance")
    st.caption("Live model advice is combined with recent ns-3 telemetry; all estimates remain separate from measured results.")
    metadata_path = ROOT / "models/live_hybrid_ns3_20261030/metadata.json"
    if not metadata_path.is_file():
        st.info("Hybrid model artifact not installed. Live control uses recent ns-3 telemetry and its trend forecast only.")
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    st.markdown(
        f"**Training file:** `{metadata['dataset_path']}`  \n"
        f"**Source:** project-generated ns-3 scenario measurements · SHA-256 `{metadata['dataset_sha256']}`  \n"
        f"**Coverage:** {metadata['rows']:,} flow measurements from {metadata['independent_scenario_groups']} scenario groups; "
        f"held out by whole scenario ({metadata['holdout_groups']} groups)."
    )
    st.caption(
        "The model predicts throughput (Mbps), delay (ms), jitter (ms), packet-loss fraction, and reception probability; "
        "QoE v2 is then calculated from those predicted KPIs. The historical production model and the separate campus dataset "
        "are not used by the live controller. This model advises only exact represented legacy-flow contexts at 35 dBm with one gNB. "
        "It does not advise NGMN gaming/video or multi-cell configurations. Unsupported contexts fall back to telemetry rules."
    )
    qoe_rows = []
    for service, scores in metadata.get("qoe_metrics", {}).items():
        if scores.get("mae") is not None:
            qoe_rows.append({"Service profile": service, "Held-out UE-flow rows": scores["rows"],
                             "Model QoE MAE": scores["mae"], "Mean-baseline QoE MAE": scores["baseline_mae"]})
    if qoe_rows:
        st.dataframe(pd.DataFrame(qoe_rows), use_container_width=True, hide_index=True)
    st.caption("The preserved production QoE model and the external 5G campus benchmark remain separate and are not loaded into live control.")
    with st.expander("Held-out KPI evaluation"):
        metrics = metadata.get("kpi_metrics", {})
        rows = [{"Target": key, **value} for key, value in metrics.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
