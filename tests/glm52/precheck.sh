#!/usr/bin/env bash
# Refuse to launch into another tenant's residue. This machine is shared, and
# an arm that launches with too little free HBM dies ninety seconds in with
# the failure buried in a rank record rather than the log.
want_gib=${1:-48}
used=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+/ 65536" | cut -d/ -f1 |
       sort -rn | head -1)
total_gib=61.27
used_gib=$(awk "BEGIN {printf \"%.1f\", $used/1024}")
free_gib=$(awk "BEGIN {printf \"%.1f\", $total_gib - $used_gib}")
echo "busiest die holds ${used_gib} GiB, free ${free_gib}, need ${want_gib}"
awk "BEGIN {exit !($free_gib >= $want_gib)}"
