from pathlib import Path
import importlib
import sys


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


def test_dashboard_modules_import_without_side_effects():
    modules = [
        "dashboard_config",
        "dashboard_data",
        "dashboard_components",
        "app",
    ]

    for module_name in modules:
        imported_module = importlib.import_module(module_name)
        assert imported_module is not None
