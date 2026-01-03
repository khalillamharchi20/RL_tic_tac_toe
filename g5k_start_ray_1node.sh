#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/RL_tic_tac_toe}"

HEAD=$(uniq "$OAR_NODEFILE" | head -n 1)
HEAD_IP=$(ssh "$HEAD" "hostname -I | awk '{print \$1}'")

echo "Single-node Ray setup"
echo "HEAD=$HEAD"
echo "HEAD_IP=$HEAD_IP"

# Stop Ray everywhere (clean start)
for N in $(uniq "$OAR_NODEFILE"); do
  ssh "$N" "
    cd $PROJECT_DIR &&
    source env/bin/activate &&
    ray stop -f >/dev/null 2>&1 || true
  "
done

# Start Ray head ONLY
ssh "$HEAD" "
  set -e
  cd $PROJECT_DIR
  source env/bin/activate
  ray start --head --node-ip-address=$HEAD_IP --port=6379 --dashboard-host=0.0.0.0
"

echo "Single-node Ray cluster started."
echo "Check with: ssh $HEAD 'source $PROJECT_DIR/env/bin/activate && ray status'"