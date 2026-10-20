#!/usr/bin/env bash
# Build the kernel as a torch extension the graph can call.
#
# Three objects: the kernel through `bisheng`, the host tiling, and the torch
# binding. The only difference from `build.sh` is `-fPIC` and the torch link
# line. ABI 1 is what this box's torch reports, and an operator project would
# force ABI 0 through its own cmake, so the two worlds cannot share an object.
set -euo pipefail
out=${1:-/tmp/routed-so}
here=$(cd "$(dirname "$0")" && pwd)
H=/opt/ascend/cann-9.2.0/aarch64-linux
SP=${SITE_PACKAGES:-/opt/trt-asc/venv/lib/python3.11/site-packages}

mkdir -p "$out"
device_incs=""
for d in asc/impl/adv_api asc/impl/basic_api asc/impl/c_api \
         asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils asc \
         asc/include asc/include/adv_api asc/include/basic_api \
         asc/include/c_api asc/include/basic_api/reg_compute \
         asc/include/simt_api asc/include/utils tikcpp/tikcfw \
         tikcpp/tikcfw/impl tikcpp/tikcfw/interface tikcpp/tikcfw/lib; do
  device_incs="$device_incs -I $H/$d"
done
device_incs="$device_incs -I /opt/ascend/cann-9.2.0/include -I $here"
host_incs="-I $H/tikcpp/tikcfw -I $H/tikcpp/tikcfw/lib -I $H/tikcpp/tikcfw/impl"
host_incs="$host_incs -I /opt/ascend/cann-9.2.0/include -I $here"
torch_incs="-I $SP/torch/include -I $SP/torch/include/torch/csrc/api/include"
torch_incs="$torch_incs -I $SP/torch_npu/include"

/opt/ascend/cann-9.2.0/bin/bisheng -xcce --npu-arch=dav-2201 -std=c++17 -O2 \
  -fPIC $device_incs -c "$here/routed_int4_kernel.cpp" \
  -o "$out/kernel.o" > "$out/kernel.log" 2>&1
g++ -std=c++17 -O2 -fPIC $host_incs -c "$here/routed_int4_tiling.cpp" \
  -o "$out/tiling.o" > "$out/tiling.log" 2>&1
g++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
  $torch_incs $host_incs -c "$here/routed_int4_binding.cpp" \
  -o "$out/binding.o" > "$out/binding.log" 2>&1
g++ -shared "$out/kernel.o" "$out/tiling.o" "$out/binding.o" \
  -o "$out/routed_int4_torch.so" \
  -L"$H/lib64" -L/opt/ascend/cann-9.2.0/lib64 \
  -L"$SP/torch/lib" -L"$SP/torch_npu/lib" \
  -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec \
  -lregister -lprofapi -ltorch -ltorch_cpu -lc10 -ltorch_npu \
  > "$out/link.log" 2>&1

[ -f "$out/routed_int4_torch.so" ] && echo "built $out/routed_int4_torch.so" || {
  echo "build failed"
  for log in kernel tiling binding link; do
    printf -- "--- %s\n" "$log"
    tail -5 "$out/$log.log" 2>/dev/null
  done
  exit 1
}
