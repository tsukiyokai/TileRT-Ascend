#!/usr/bin/env bash
# Two arms, alternated inside one session: A B A B A B.
#
# Running one arm after the other does not work at this scale. Two identical
# configurations measured in successive chains differed by 0.201 ms in the
# round's tenth percentile, which is the size of the effects being looked for.
# Alternating pairs each A with a B that ran under the same conditions, so the
# drift differences out.
#
# Usage: chain_alternating.sh <runner> <arm-a-name> <arm-b-name> [b-overlay]
set -uo pipefail
runner=$1
name_a=$2
name_b=$3
overlay=${4:-}
T=/workspace/experiments/kernelpatch-20261019

for pass in 1 2 3; do
  bash "$runner" "$name_a-$pass"
  [ -f "$T/$name_a-$pass/rank-0.json" ] || { echo "Error: $name_a-$pass empty"; exit 1; }
  bash "$runner" "$name_b-$pass" "$overlay"
  [ -f "$T/$name_b-$pass/rank-0.json" ] || { echo "Error: $name_b-$pass empty"; exit 1; }
done
echo "=== alternating chain done $(date -Is)"
