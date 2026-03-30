#!/bin/bash
apt-get install -y git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
python run.py --demo --no-browser --port ${PORT:-5000}
