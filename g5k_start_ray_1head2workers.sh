#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/RL_tic_tac_toe}"

# Load HEAD/W1/W2/HEAD_IP
source ./g5k_nodes.sh

echo "Starting Ray head on $HEAD (IP: $HEAD_IP)"
ssh "$HEAD" "
  set -e
  cd $PROJECT_DIR
  source env/bin/activate
  ray stop -f >/dev/null 2>&1 || true
  ray start --head --node-ip-address=$HEAD_IP --port=6379 --dashboard-host=0.0.0.0
"

for W in "$W1" "$W2"; do
  echo "Starting Ray worker on $W"
  ssh "$W" "
    set -e
    cd $PROJECT_DIR
    source env/bin/activate
    ray stop -f >/dev/null 2>&1 || true
    ray start --address=$HEAD_IP:6379
  "
done

echo "Cluster started."
echo "Check with: ssh $HEAD 'source $PROJECT_DIR/env/bin/activate && ray status'"
