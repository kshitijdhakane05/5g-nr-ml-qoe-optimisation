from __future__ import annotations

import unittest
from pathlib import Path
import sys


class LiveNs3DashboardSmokeTests(unittest.TestCase):
    def test_live_page_renders_real_time_controls_and_metrics_without_errors(self):
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
            self.assertIn("Live network simulation", [x.value for x in app.title])
            self.assertIn("📡 5G QoE Lab", [x.value for x in app.title])
            self.assertIn("Start live simulation", [x.label for x in app.button])
            self.assertGreaterEqual(len(app.get("plotly_chart")), 4)
            for graph_heading in ("QoE trend", "Throughput", "Delay and jitter", "Packet loss"):
                self.assertIn(graph_heading, [x.value for x in app.subheader])
            streamlit_config = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
            self.assertIn('base = "dark"', streamlit_config.read_text())
            self.assertTrue(any("QoE target" == m.label for m in app.slider))
            self.assertTrue(any("QoE score and measurement units" == e.label for e in app.expander))
            self.assertTrue(any(h.value in {"Active network and traffic settings", "Last verified network settings"}
                                for h in app.subheader))
            self.assertTrue(any("Live UE topology and state" == h.value for h in app.subheader))
            self.assertTrue(any("Select a UE marker on the topology" in c.value for c in app.caption))
            self.assertTrue(any(h.value in {"Live network feed", "Recent measurement history"} for h in app.subheader))
            self.assertTrue(any("Live controller and action feed" == h.value for h in app.subheader))
            def navigation():
                return next(x for x in app.radio if x.label == "Workspace")

            self.assertEqual(navigation().options, [
                "Run simulation", "Overview", "Experiments", "Analytics", "Tools",
            ])
            mode = next(x for x in app.selectbox if x.label == "Controller mode")
            self.assertEqual(mode.value, "Recommend")
            self.assertTrue(any(x.label == "Band 1 bandwidth · startup only" for x in app.selectbox))
            self.assertTrue(any(x.label == "Best-effort packet rate · packets/s per UE" for x in app.number_input))
            self.assertTrue(any(x.label == "Low-latency packet rate · packets/s per UE" for x in app.number_input))

            navigation().set_value("Overview").run(timeout=30)
            self.assertIn("Network overview", [x.value for x in app.title])
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            navigation().set_value("Experiments").run(timeout=30)
            self.assertIn("Experiments", [x.value for x in app.title])
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            navigation().set_value("Analytics").run(timeout=30)
            self.assertIn("Analytics", [x.value for x in app.title])
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            navigation().set_value("Tools").run(timeout=30)
            tools_box = next(x for x in app.selectbox if x.label == "Additional tools")
            self.assertIn("🤖 QoE Predictor (offline dataset)", tools_box.options)
            self.assertIn("⚙️ Network Optimiser (offline analysis)", tools_box.options)
            self.assertIn("🔁 Closed-Loop Verification (single run)", tools_box.options)
            self.assertIn("🧠 Live Hybrid Model & Dataset Provenance", tools_box.options)
            self.assertIn("🌐 External 5G Campus Dataset Benchmark", tools_box.options)
            for option, expected_title in (
                ("🔍 Simulation Explorer", "🔍 Simulation Data Explorer"),
                ("🤖 QoE Predictor (offline dataset)", "🤖 QoE Predictor · Offline dataset"),
                ("⚙️ Network Optimiser (offline analysis)", "⚙️ Network Optimiser · Offline analysis"),
                ("🔁 Closed-Loop Verification (single run)", "🔁 Closed-Loop Verification · Single run"),
                ("🧪 Experimental KPI Results (offline)", "🧪 Experimental KPI results (offline)"),
                ("🧠 Live Hybrid Model & Dataset Provenance", "Live Hybrid Model & Dataset Provenance"),
                ("🌐 External 5G Campus Dataset Benchmark", "External 5G Campus Dataset Benchmark"),
            ):
                next(x for x in app.selectbox if x.label == "Additional tools").set_value(option).run(timeout=30)
                self.assertIn(expected_title, [x.value for x in app.title])
                self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

            overview_link = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            overview_link.query_params["view"] = "overview"
            overview_link.run(timeout=30)
            self.assertEqual(len(overview_link.exception), 0, [e.message for e in overview_link.exception])
            self.assertIn("Network overview", [x.value for x in overview_link.title])
        finally:
            for module_name in list(sys.modules):
                if (module_name == "streamlit" or module_name.startswith("streamlit.")
                        or module_name.startswith("dashboard.")):
                    sys.modules.pop(module_name, None)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
