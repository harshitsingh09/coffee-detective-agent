from pathlib import Path
from unittest import TestCase

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(TestCase):
    def test_app_renders_all_portfolio_sections_without_exception(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path).run(timeout=30)

        self.assertEqual([item.value for item in app.exception], [])
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Investigate", "Scenario Lab", "How It Works"],
        )
