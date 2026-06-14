#!/usr/bin/env bash
# One-command setup: creates a virtualenv and installs pinned deps.
# Usage:  ./setup.sh   then   source .venv/bin/activate && streamlit run app.py
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "→ Creating virtual environment in .venv"
"$PYTHON" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "→ Upgrading pip"
python -m pip install --upgrade pip >/dev/null

echo "→ Installing pinned dependencies (this can take a few minutes)"
pip install -r requirements.txt

echo
echo "✅ Done. Next steps:"
echo "   source .venv/bin/activate"
echo "   python train.py        # creates export.pkl (skip if you already have one)"
echo "   streamlit run app.py"
