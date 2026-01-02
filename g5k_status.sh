#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/public/RL_tic_tac_toe}"
source ./g5k_nodes.sh

ssh "$HEAD" "
  set -e
  source $PROJECT_DIR/env/bin/activate
  ray status
"
