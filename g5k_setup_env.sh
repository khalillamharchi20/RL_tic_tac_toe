#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/RL_tic_tac_toe}"

echo "Project dir: $PROJECT_DIR"
echo "Nodes:"
uniq "$OAR_NODEFILE" | sed 's/^/ - /'

for N in $(uniq "$OAR_NODEFILE"); do
  echo "==> Setting up on $N"
  ssh "$N" "
    set -e
    cd $PROJECT_DIR
    python3 -m venv env
    source env/bin/activate
    pip install -U pip
    pip install ray torch numpy matplotlib
  "
done

echo "Done."
