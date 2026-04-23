"""Entrypoint for hosting platforms: launch the Streamlit app.

This script runs `python -m streamlit run streamlit_toxicity_app.py` and respects
an optional PORT environment variable (defaults to 8501). It attempts to
replace the current process with the Streamlit process for cleaner signal
handling; falls back to subprocess.run if exec fails.
"""

import os
import subprocess
import sys


def main():
    port = os.environ.get("PORT", "8501")
    app = os.environ.get("STREAMLIT_APP", "streamlit_toxicity_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(port), "--server.address", "0.0.0.0"]
    try:
        os.execvp(cmd[0], cmd)
    except Exception:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
