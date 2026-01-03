#!/usr/bin/env bash
set -euo pipefail

for N in $(uniq "$OAR_NODEFILE"); do
  echo "==> ray stop on $N"
  ssh "$N" "ray stop -f >/dev/null 2>&1 || true"
done

echo "Done."
