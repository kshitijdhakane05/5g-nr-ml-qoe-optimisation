# External 5G campus dataset evaluation

## Decision and scope

The project now contains a reproducible, separate benchmark pipeline for the run-level throughput files in Zenodo record [10.5281/zenodo.13754300](https://doi.org/10.5281/zenodo.13754300), version v1 (retrieved 2026-10-29). The two selected files are stored unchanged in `data/raw/5g_campus_network_qos_zenodo_13754300/`. The four multi-gigabyte packet-trace files were not downloaded; they are not inputs to this run-level benchmark. Zenodo lists the record under CC-BY-4.0.

This is not a production model migration. The external data does not cover the project’s UE count, transmit power, scenario size, traffic classes, or offered packet rates. Its `bw`, DL slots, DL:UL ratio, gNB implementation, and B200 SDR have no verified equivalent in `ns-3.48/scratch/qoe-dataset.cc`. Mapping them into the existing optimizer would claim support that the simulator does not provide. The production model, training path, dashboard, and optimizer therefore remain on their existing baseline until compatible ns-3 training data is collected. No hidden fallback was added.

## Provenance

- Source: Raffeck, Simon; Grøsvik, Sebastian; Lange, Stanislav; Hoßfeld, Tobias; Zinner, Thomas; Geißler, Stefan. *5G Campus Network QoS Dataset for Open-Source gNB Implementations*. Zenodo, v1, published 2024-09-12. DOI: [10.5281/zenodo.13754300](https://doi.org/10.5281/zenodo.13754300).
- Associated paper: Raffeck et al., *Parameterizing 5G New Radio: A Comparative Measurement Study on Throughput and Delay*, CNSM 2024, DOI [10.23919/CNSM62983.2024.10814615](https://doi.org/10.23919/CNSM62983.2024.10814615).
- License: CC-BY-4.0 (attribution required).
- Selected source-file checksums (Zenodo MD5 / local SHA-256):
  - `ntnu_tput_all_Throughput.csv`: `9e30456494de6d31e017fb00fef44478` / `c4968908d33aadcf8d1b5b8030db679309dbd11b745c1933bd36d56ed55f715f`
  - `wue_tput_all_Throughput.csv`: `b62e0f47b796e73d3f8ae44eca40937f` / `dfa1f5bbff178ddc44407453dac146b2d6ab18332a214d61015c9a9b45a4f15b`

## Data audit and mapping

The selected files contain 142 NTNU and 160 WUE rows (302 total), one row per repeated run. They have no missing values or exact duplicates. Both contain 16 source columns, including an unnamed row index, six observed outcome fields (`mbpsactual_*`, `meanjitterms_*`, `meanloss_*`), five configuration fields (`gnb`, `sdr`, `bw`, `slots`, `ratio`), offered uplink/downlink rates, repetition, and scenario label. Across the combined data: gNB is OAI or srsRAN; SDR is only B200; bandwidth is 20 or 40; slots are 5, 10, or 20; ratio is 1, 2, or 4. Offered rates are 25/50 uplink and 75/150 downlink in the source's `mbps` fields.

| Role | Fields | Use |
| --- | --- | --- |
| Configuration inputs | `gnb`, `bw`, `slots`, `ratio`, `mbpsoffered_uplink`, `mbpsoffered_downlink` | Benchmark features; `sdr` is constant and excluded |
| Context | `site` (`ntnu`/`wue`) | Benchmark feature; no mapping to the ns-3 topology |
| Measured targets | actual uplink/downlink throughput, mean uplink/downlink jitter, mean uplink/downlink loss | Six separate regressors |
| Identifiers/metadata | source row index, `rep`, `scenario`, source file | Excluded from features; repetitions grouped for validation |
| Derived QoE | None | No human QoE label or defensible project QoE proxy is supplied |

The source metadata describes actual/offered throughput in “MBps” while field names and offered rates use `mbps`; values and rate labels do not resolve this discrepancy. The loss fields are not assigned a clear percentage or fraction unit. The benchmark preserves these targets in source-reported numeric units and does not relabel or convert them. The two selected files cover throughput, jitter, and loss, but not the packet-level one-way-delay traces.

## Preprocessing and evaluation

Run `/tmp/5g-project-ml-venv/bin/python -m ml.campus_dataset` to validate required columns, missingness, duplicates, and numeric fields; preserve raw rows and create the external-domain model and reports under `models/campus_5g_zenodo_13754300/`. Features are the seven listed configuration/context inputs. Source row index, repetition, scenario, and outcomes are excluded from features. All repeated rows sharing the same gNB, bandwidth, slot count, DL:UL ratio, and offered rates are assigned to one group across both testbeds. Five-fold GroupKFold estimates performance on configuration groups not seen in training.

Four models were compared for each target: mean dummy, scaled Ridge, random forest, and histogram gradient boosting. `evaluation.csv` contains pooled grouped out-of-fold MAE, RMSE, and R²; `grouped_cv_predictions.csv` contains corresponding out-of-fold random-forest estimates. These are internal benchmark scores, not ns-3 validation. MAE/RMSE remain in source-reported numeric units. There are only 32 distinct grouped settings, and 5-fold scores do not establish deployment reliability. Jitter downlink did not beat the mean baseline by R²; measured evidence does not support using that model for ranking. Other targets show varying held-out signal, but remain subject to unit ambiguity and domain mismatch.

The artifact's `predict_target` interface emits domain and marginal range warnings. Marginal range checks do not establish joint support or uncertainty; callers must not treat this interface as ns-3 inference.

## ns-3 execution check

The project runner completed two real runs with the same scenario ID, UE count, load, power, placement, and duration; bandwidth was the only changed network parameter (20 vs 50 MHz). Both runs wrote their own configuration, command, summary, and measurement CSV under `results/verification/campus_ns3_mapping_check_20261029/matched_seed/`. The 20 MHz run received zero packets on both flows. The 50 MHz run received 5,998/6,000 and 2,997/3,000 packets, with per-flow throughput 10.2366 and 51.1488 Mbps. This verifies the existing bandwidth-to-ns-3 mapping and demonstrates a measurable simulator difference for these settings. It does not validate or use the external campus model.

## Alternatives assessed

The [RANalyzer dataset](https://github.com/wineslab/RANalyzer-Dataset) reports 9,430 end-to-end OAI test runs from 69 software releases, with fixed 60 MHz channel, n78, X410 SDR setup and varying target bitrate. Its CC-BY-4.0 license and rich measurements make it a useful software-regression dataset, but those fixed radio conditions offer less configuration diversity for this optimizer. It was not mixed with the campus dataset because the domains and definitions differ.

## Baseline and rollback

The original 720-line generated source is preserved at `data/backup/qoe_dataset_original.csv` (SHA-256 `2a4bd15fd752be63cee31c7d0920de078980a0a55287f69da1a985c51dd186f8`). The original production artifact and evaluation outputs are copied, checksum-verified under `models/baseline_20261029/` and `results/ml/baseline_20261029/`. The production files were not replaced. Rollback is therefore the current behavior: keep using `models/qoe_model.joblib` with the existing training source. Do not promote `models/campus_5g_zenodo_13754300/model.joblib` into production until its inputs, target units, and simulator mapping are resolved and validated against actual ns-3 runs.
