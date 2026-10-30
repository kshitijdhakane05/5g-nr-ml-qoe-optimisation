from pathlib import Path
import unittest

import joblib
import pandas as pd

from ml.campus_dataset import DATA_DIR, FEATURES, TARGETS, load_dataset, predict_target


class CampusDatasetTests(unittest.TestCase):
    def test_external_dataset_schema_and_provenance(self):
        frame = load_dataset(DATA_DIR)
        self.assertEqual(len(frame), 302)
        self.assertTrue(frame[FEATURES + TARGETS].notna().all().all())
        self.assertEqual(frame["configuration_group"].nunique(), 32)
        self.assertNotIn("scenario", FEATURES)
        self.assertNotIn("rep", FEATURES)

    def test_trained_artifact_loads_and_marks_domain(self):
        artifact = Path(__file__).resolve().parents[1] / "models/campus_5g_zenodo_13754300/model.joblib"
        bundle = joblib.load(artifact)
        sample = load_dataset(DATA_DIR).iloc[[0]][FEATURES]
        result = predict_target(bundle, sample, "mbpsactual_downlink")
        self.assertEqual(len(result["prediction"]), 1)
        self.assertTrue(result["domain"].endswith("not ns-3"))
        outside = sample.copy()
        outside["bw"] = 80
        result = predict_target(bundle, outside, "mbpsactual_downlink")
        self.assertEqual(result["support_status"], "outside_training_range")
