#!/usr/bin/env bash
set -euo pipefail

NODES=($(uniq "$OAR_NODEFILE"))
if [ "${#NODES[@]}" -lt 3 ]; then
  echo "Need at least 3 nodes in OAR reservation, got ${#NODES[@]}"
  exit 1
fi

HEAD="${NODES[0]}"
W1="${NODES[1]}"
W2="${NODES[2]}"

HEAD_IP=$(ssh "$HEAD" "hostname -I | awk '{print \$1}'")

echo "HEAD=$HEAD"
echo "W1=$W1"
echo "W2=$W2"
echo "HEAD_IP=$HEAD_IP"

export HEAD W1 W2 HEAD_IP
