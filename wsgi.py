"""WSGI entrypoint for production servers.

This avoids import-name conflicts between the top-level `app.py` file and the
`app/` package directory.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

APP_FILE = Path(__file__).with_name("app.py")
_spec = spec_from_file_location("app_main", APP_FILE)
if _spec is None or _spec.loader is None:
	raise RuntimeError(f"Unable to load WSGI app from {APP_FILE}")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app
