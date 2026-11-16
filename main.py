"""Unified entrypoint for Route.it API & Web Dashboard.
Runs Uvicorn listening on 0.0.0.0:$PORT (default 8000).
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in ["engine", "reference", "roads", "traffic", "backend", "."]:
    _p_str = str((_ROOT / _p).resolve())
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
