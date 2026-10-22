#!/usr/bin/env bash
# One whole-machine arm inside the unified image, which carries the engine
# stack and CANN's op compiler itself. It replaces `activate.sh`, which lived
# in the deleted container's writable layer.
#
#   TAG      names the arm; its cache root and port range come from it
#   WIDTH    verify width
#   TREE     the source checkout, mounted read-only
#   OUT      where the records go
#   MODEL    the checkpoint, read-only
#   CAPTURE  the 8K prefix case
#   remaining arguments go to the bench
#
# Exit 2 means the gate refused, so a driver can wait rather than consume it.
set -uo pipefail
tag=${TAG:?}
width=${WIDTH:?}
tree=$(realpath "${TREE:?}")
out_root=$(realpath -m "${OUT:?}")
out=$out_root/$tag
model=${MODEL:?}
capture=${CAPTURE:?}
utilisation=${UTILISATION:-0.78}
attention=${ATTENTION:-sfa}

source /opt/ascend/cann-9.2.0/set_env.sh
# The engine stack comes from the pinned trees under RUNTIME, because the
# omniinfer image ships `omni_npu` rather than `vllm_ascend` and the unified
# image carries only vllm's dist-info -- its version string ends in `.empty`,
# which `importlib.metadata.version` reports happily and `import` does not.
#
# Append, never assign: set_env.sh puts CANN's `tbe` on PYTHONPATH and GE
# imports it, so assigning turns that into `OpCompileProcessor init failed`.
runtime=${RUNTIME:?}
# `vendor-metadata` carries vllm_ascend's egg-info, and the egg-info is the only
# thing that declares `ascend = vllm_ascend:register` under
# `vllm.platform_plugins`. Without it `VLLM_PLUGINS=ascend` selects nothing, no
# platform registers, and the engine dies in `DeviceConfig` on an empty device
# string rather than on anything to do with devices.
export PYTHONPATH="$tree:$runtime/vendor-hook-pinned/vllm-a982-py311-compatible:$runtime/vendor-hook-pinned/ascend-python-build:$runtime/vendor-hook-pinned/vendor-metadata:$runtime/vendor-hook-deps/transformers519:$runtime/vendor-hook-deps/python${PYTHONPATH:+:$PYTHONPATH}"

# `fla_npu` cannot be dropped even though its kernels are not wanted here:
# `vllm_ascend/ops/gdn.py` imports it unconditionally for Qwen3.5. Its
# `libcust_opapi.so` needs GLIBCXX_3.4.29, which neither the image's
# libstdc++ 6.0.28 nor the host's has; the image's gcc-toolset-12 does.
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Only the ascend plugin: the image's own omni_npu registers against vllm 0.30's
# entrypoint layout and the pinned vllm is 0.25.1.
export VLLM_PLUGINS=ascend
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
export XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/triton
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# A cache root per arm, because a shared one replays another arm's compiled
# graph, and a port range per arm, because two arms of one pass cannot share it.
export VLLM_CACHE_ROOT=$out_root/cache-$tag
export HCCL_NPU_SOCKET_PORT_RANGE="${PORT_BASE:-27000}-$((${PORT_BASE:-27000} + 19))"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

rm -rf "$out" "$VLLM_CACHE_ROOT"
mkdir -p "$out" "$VLLM_CACHE_ROOT"
echo "=== $tag width=$width util=$utilisation hccl=$HCCL_NPU_SOCKET_PORT_RANGE $* $(date -Is)"

# Static kernel packages are cached below the working directory.
cd "$out"
timeout "${ARM_TIMEOUT:-14400}" torchrun --nproc-per-node 16 \
  --master-port $((51000 + RANDOM % 2000)) \
  "$tree/tests/glm52/bench_cann_mtp.py" \
  --checkpoint "$model" \
  --capture-case "$capture" \
  --source-commit "$tag" \
  --max-tokens 1024 --capacity 9344 --warmup 3 --requests 3 \
  --width "$width" --output "$out" \
  --gpu-memory-utilization "$utilisation" \
  --target-graph --compile-target \
  --super-kernel --attention-kernel "$attention" --fp32-head "$@" \
  > "$out/launch.log" 2>&1
status=$?
echo "=== $tag exit $status $(date -Is)"
exit $status
