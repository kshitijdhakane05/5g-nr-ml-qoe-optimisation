from __future__ import annotations

import unittest
from pathlib import Path

from dashboard.branding import PROJECT_DISPLAY_NAME


class DashboardBrandingTests(unittest.TestCase):
    def test_product_name_is_shared_by_page_title_and_sidebar(self):
        project_root = Path(__file__).resolve().parents[1]
        app_source = (project_root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(PROJECT_DISPLAY_NAME, "5G QoE Lab")
        self.assertIn('page_title=f"{PROJECT_DISPLAY_NAME} · Live ns-3 Simulator"', app_source)
        self.assertIn('st.title(f"📡 {PROJECT_DISPLAY_NAME}")', app_source)


if __name__ == "__main__":
    unittest.main()
