"""Deployment-friendly entrypoint.

Many hosts look for a top-level `app.py`. This file imports and calls
`main.main()` from the existing main.py. If that import fails it falls
back to launching Streamlit directly so the app still starts.
"""

import os
import sys
import subprocess

try:
    # Prefer calling the existing main() entrypoint
    from main import main as start
except Exception:
    def start():
        port = os.environ.get("PORT", "8501")
        app = os.environ.get("STREAMLIT_APP", "streamlit_toxicity_app.py")
        cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(port), "--server.address", "0.0.0.0"]
        subprocess.run(cmd, check=True)


# Expose a module-level callable named `app` for platform detection.
# Many hosts look for a callable/variable named `app` in app.py.
app = start


if __name__ == "__main__":
    start()
