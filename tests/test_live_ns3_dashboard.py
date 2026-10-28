from __future__ import annotations

import unittest
from pathlib import Path
import sys


class LiveNs3DashboardSmokeTests(unittest.TestCase):
    def test_live_page_renders_controls_and_qoe_chart_without_errors(self):
        # test_dashboard installs a Streamlit stub and imports dashboard modules
        # against it. AppTest needs both the real package and fresh UI modules.
        saved_modules = {
            name: module for name, module in list(sys.modules.items())
            if name == "streamlit" or name.startswith("streamlit.") or name.startswith("dashboard.")
        }
        for module_name in list(saved_modules):
            sys.modules.pop(module_name, None)
        try:
            from streamlit.testing.v1 import AppTest

            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            self.assertIn("Live Simulator", [x.value for x in app.title])
            self.assertIn("📡 [Project Name]", [x.value for x in app.title])
            self.assertIn("Start ns-3 simulation/controller", [x.label for x in app.button])
            self.assertGreaterEqual(len(app.get("plotly_chart")), 1)
            self.assertTrue(any("Target simulated network quality score (0–1)" == m.label for m in app.slider))
            def navigation():
                return next(x for x in app.radio if x.label == "Main navigation")

            self.assertEqual(navigation().options, [
                "🛰️ Live Simulator", "📊 Overview", "🧪 Experiments", "📈 Analytics", "🧰 More tools",
            ])
            preset_box = next(x for x in app.selectbox if x.label == "Preset test scenario")
            self.assertIn("Baseline", preset_box.options)
            self.assertIn("Current session", [x.value for x in app.subheader])
            self.assertIn("Scenario configuration", [x.value for x in app.subheader])
            self.assertIn("Simulation controls", [x.value for x in app.subheader])
            checkbox_labels = [x.label for x in app.checkbox]
            for label in (
                "Allow exploratory configurations",
                "Advanced override: permit high-risk configurations",
                "Allow repeat trials of configurations already attempted",
                "Run remaining experimental candidates after epoch limit",
            ):
                self.assertIn(label, checkbox_labels)
            number_input_labels = [x.label for x in app.number_input]
            self.assertIn("Maximum number of normal simulation runs", number_input_labels)
            self.assertIn("Normal adaptive action budget", number_input_labels)
            self.assertIn("Per-run timeout (seconds)", number_input_labels)
            config_mode = next(x for x in app.radio if x.label == "Scenario configuration")
            config_mode.set_value("Custom configuration").run(timeout=30)
            self.assertTrue(any(x.label == "Band 2 bandwidth (MHz)" for x in app.selectbox))
            self.assertTrue(any(x.label == "Best-effort rate (packets/s)" for x in app.number_input))

            navigation().set_value("📊 Overview").run(timeout=30)
            self.assertIn("Network overview", [x.value for x in app.title])
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            navigation().set_value("🧪 Experiments").run(timeout=30)
            self.assertIn("Experiments", [x.value for x in app.title])
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            navigation().set_value("📈 Analytics").run(timeout=30)
            self.assertIn("Analytics", [x.value for x in app.title])
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            navigation().set_value("🧰 More tools").run(timeout=30)
            tools_box = next(x for x in app.selectbox if x.label == "Supporting tools")
            self.assertIn("🤖 QoE Predictor", tools_box.options)
            self.assertIn("⚙️ Network Optimiser", tools_box.options)
            self.assertIn("🔁 Closed-Loop Verification", tools_box.options)
            for option, expected_title in (
                ("🔍 Simulation Explorer", "🔍 Simulation Data Explorer"),
                ("🤖 QoE Predictor", "🤖 QoE Predictor"),
                ("⚙️ Network Optimiser", "⚙️ Network Optimiser"),
                ("🔁 Closed-Loop Verification", "🔁 Closed-Loop Verification"),
                ("🧪 Experimental KPI Results (offline)", "🧪 Experimental KPI results (offline)"),
            ):
                next(x for x in app.selectbox if x.label == "Supporting tools").set_value(option).run(timeout=30)
                self.assertIn(expected_title, [x.value for x in app.title])
                self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        finally:
            for module_name in list(sys.modules):
                if (module_name == "streamlit" or module_name.startswith("streamlit.")
                        or module_name.startswith("dashboard.")):
                    sys.modules.pop(module_name, None)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
