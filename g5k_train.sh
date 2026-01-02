#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/public/RL_tic_tac_toe}"
source ./g5k_nodes.sh

NUM_WORKERS="${1:-8}"
ITERS="${2:-200}"
EPW="${3:-50}"

echo "Training on HEAD=$HEAD with num_workers=$NUM_WORKERS iters=$ITERS episodes_per_worker=$EPW"

ssh "$HEAD" "
  set -e
  cd $PROJECT_DIR
  source env/bin/activate
  python main.py --ray-address auto --num-workers $NUM_WORKERS --iters $ITERS --episodes-per-worker $EPW
"
